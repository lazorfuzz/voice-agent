"""Frequently-used-commands tracker for the web UI's home-screen pills.

A "command" is any chat turn that actually triggered a tool call — the tool signal is the
classifier, so chitchat never pollutes the list. Commands are normalized (case, punctuation,
courtesy fillers) so "Turn on the lights!" and "turn on the lights please" aggregate, and
ranked by a decayed-frequency score (half-life ~2 weeks) so stale habits fade out without
keeping any history. Context-dependent turns ("turn them off", "stop") are skipped — a pill
must make sense with no conversation behind it.

Store: commands_freq.json next to this file (gitignored — it's personal usage data).

Two processes write here — the token server (text chat) and the agent worker (voice) — so
every read-modify-write happens under an fcntl file lock with a reload from disk inside it.
"""
import fcntl
import json
import os
import re
import threading
import time

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands_freq.json")
_HALF_LIFE = 14 * 86400          # frecency half-life: two weeks
_MAX_LEN = 60                     # longer than this isn't a "command", it's a message
_MAX_ENTRIES = 40                 # prune cap
_MIN_SCORE = 0.05                 # prune floor (a single use fades out in ~9 weeks)

# courtesy tokens stripped from the edges (loop, so "can you please ..." fully strips).
# Voice transcripts often open with the wake word / assistant name ("hey kronik ...") —
# fold the configured ones in so voice and chat phrasings of a command aggregate.
_LEAD = {"please", "hey", "ok", "okay", "yo", "now", "can", "could", "would", "you", "u",
         "kronik", "also", "and", "then", "just", "go", "ahead"}
_LEAD |= {w for k in ("WAKE_WORD", "ASSISTANT_NAME")
          for w in os.environ.get(k, "").lower().split() if w}
_TAIL = {"please", "now", "thanks", "thank", "you", "u", "rn", "again", "tho", "though",
         "right", "currently", "atm"}   # "right" catches the trailing "right now" idiom
# a pill must be self-contained: skip turns that lean on conversation context
_CONTEXTUAL = re.compile(r"\b(it|them|that|this|those|these|him|her|one|ones|there)\b")

_lock = threading.Lock()          # thread-level; the fcntl lock below is process-level


def _read():
    """Fresh read from disk — the other process may have written since we last looked."""
    try:
        with open(_PATH) as f:
            data = json.load(f)
        assert data.get("version") == 1 and isinstance(data.get("commands"), dict)
        return data
    except Exception:
        return {"version": 1, "commands": {}}


def _write(store):
    tmp = _PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f)
    os.replace(tmp, _PATH)        # atomic — a crash never corrupts the store


class _flocked:
    """Exclusive cross-process lock so token server + agent worker never lose updates."""
    def __enter__(self):
        self._f = open(_PATH + ".lock", "w")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()
        return False


def _normalize(text):
    """Aggregation key, or None if this turn isn't pill material."""
    t = re.sub(r"[^\w\s%$°'-]", " ", (text or "").lower())
    words = t.split()
    while words and words[0] in _LEAD:
        words.pop(0)
    while words and words[-1] in _TAIL:
        words.pop()
    if len(words) < 2:            # "stop", "again" — context-dependent
        return None
    key = " ".join(words)
    if len(key) > _MAX_LEN or _CONTEXTUAL.search(key):
        return None
    return key


def _decayed(score, last, now):
    return score * (0.5 ** ((now - last) / _HALF_LIFE))


def record(text, tools):
    """Count one completed command turn. Call only when the turn ran ≥1 tool."""
    if not tools:
        return
    key = _normalize(text)
    if not key:
        return
    now = time.time()
    with _lock, _flocked():
        store = _read()           # re-read under the lock: never clobber the other writer
        cmds = store["commands"]
        e = cmds.get(key)
        if e:
            e["score"] = _decayed(e["score"], e["last"], now) + 1.0
            e["last"] = now
            e["tools"] = sorted(set(e.get("tools", [])) | set(tools))
        else:
            # the normalized key IS the pill text: fillers/wake-word stripped, self-contained
            cmds[key] = {"score": 1.0, "last": now, "tools": sorted(set(tools))}
        # prune: drop faded entries, cap total
        for k in [k for k, v in cmds.items() if _decayed(v["score"], v["last"], now) < _MIN_SCORE]:
            del cmds[k]
        if len(cmds) > _MAX_ENTRIES:
            for k, _ in sorted(cmds.items(),
                               key=lambda kv: _decayed(kv[1]["score"], kv[1]["last"], now))[:len(cmds) - _MAX_ENTRIES]:
                del cmds[k]
        _write(store)


def top(n=8):
    """Top commands by decayed score: [{text, tools}], best first. Always reads fresh —
    the file is tiny and the other process may have just recorded a voice command."""
    now = time.time()
    cmds = _read()["commands"]
    ranked = sorted(cmds.items(),
                    key=lambda kv: _decayed(kv[1]["score"], kv[1]["last"], now), reverse=True)
    return [{"text": k[:1].upper() + k[1:], "tools": e.get("tools", [])}
            for k, e in ranked[:n]]

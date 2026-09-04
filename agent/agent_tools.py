"""Backend for Kronik's voice tools: dispatch/resume/query local opencode agents.
All dispatches are DETACHED (non-blocking) so voice stays responsive; results land
in the sessions DB and are read back on demand."""
import os, re, sys, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_db as db

# Repo root = the dir above agent/ — derived from THIS file, never from cwd or a hardcoded
# ~/voice-agent, so the stack works from a clone in any directory.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(BASE, "sessions_ws")
PY = os.path.join(BASE, ".venv/bin/python")
DAEMON = os.path.join(BASE, "scripts", "daemon.py")
RUNNER = os.path.join(BASE, "agent", "run_session.py")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "session"


def _launch(name, workdir, sid_arg, message):
    """Fire the detached runner (survives on its own; DB updated on completion)."""
    os.makedirs(workdir, exist_ok=True)
    logf = os.path.join(workdir, "_run.log")
    subprocess.Popen(
        [PY, DAEMON, logf, PY, RUNNER, name, workdir, sid_arg, message],
        start_new_session=True,
    )


def dispatch(name: str, task: str) -> str:
    workdir = os.path.join(WS, _slug(name))
    db.create(name, task, workdir)
    _launch(name, workdir, "NEW", task)
    # NOTE: this string doubles as a user-facing reply (chat's loop-guard can speak it
    # verbatim) — keep it natural and mechanics-free: no "agent"/"session" talk.
    return "On it — I'm looking into that now."


def _queue_when_ready(name: str, message: str, timeout: float = 300.0):
    """Spin-up race: a follow-up can arrive before the session's FIRST turn has recorded its
    opencode session id. Dropping it (or telling the model to 'retry later' — it has no
    timer) silently loses the user's question, so park a daemon thread that waits for the id
    and then launches the turn. Worker restart loses the queued message — acceptable."""
    import threading, time as _time

    def _wait():
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            s = db.get(name)
            if not s:
                return                       # session deleted meanwhile
            if s.get("opencode_session_id"):
                db.set_running(name, message)
                _launch(name, s["workdir"], s["opencode_session_id"], message)
                return
            _time.sleep(2.0)

    threading.Thread(target=_wait, daemon=True).start()


def send_message(name: str, message: str) -> str:
    s = db.get(name)
    if not s:
        return f"(internal: no session named '{name}' — start one with run_agent instead)"
    if not s.get("opencode_session_id"):
        _queue_when_ready(name, message)     # sends itself once the first turn lands
        return "On it — I'm digging into that now."
    db.set_running(name, message)
    _launch(name, s["workdir"], s["opencode_session_id"], message)
    return "On it — I'm digging into that now."


def list_recent(limit: int = 6):
    return db.list_recent(limit)


def get_state(name: str):
    return db.get(name)


def delete_session(name: str) -> str:
    existed = db.delete(name)
    return (f"Deleted session '{name}' and its files."
            if existed else f"I don't have a session called '{name}'.")

"""Detached worker: run one opencode turn (new or resumed), capture the session id
and clean output, and write the result back to the sessions DB.
Usage: run_session.py <name> <workdir> <NEW|session_id> <message>"""
import sys, os, re, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_db as db

# Model for dispatched sessions. Empty -> opencode uses its own configured default. Set
# OPENCODE_MODEL to override (use the SAME model as the voice agent so the box doesn't load a
# second 35B model and thrash RAM). --agent local (below) gives them the lean tool-forward prompt.
MODEL = os.environ.get("OPENCODE_MODEL", "").strip()
OC = os.path.expanduser("~/.opencode/bin/opencode")


def clean(raw: str) -> str:
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)              # strip ANSI
    lines = [
        l for l in raw.splitlines()
        if not l.startswith("timestamp=") and not l.lstrip().startswith("> ")
    ]
    return "\n".join(l for l in lines if l.strip()).strip()


def main():
    name, workdir, sid_arg, message = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(workdir, exist_ok=True)

    cmd = [OC, "run", "--auto", "--agent", "local", "--dir", workdir,
           "--print-logs", "--log-level", "INFO"]
    if MODEL:
        cmd += ["-m", MODEL]
    if sid_arg and sid_arg != "NEW":
        cmd += ["-s", sid_arg]
    cmd += [message]

    try:
        # cwd + --dir + PWD all set to the per-session workdir: cwd/--dir set the real dir,
        # PWD overrides the stale value inherited from the worker (opencode's shell reads $PWD).
        env = {**os.environ, "PWD": workdir}
        p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=1200)
        raw, log = p.stdout, p.stderr
        status = "done" if p.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        raw, log, status = "", "", "error"
    except Exception as e:
        raw, log, status = "", str(e), "error"

    if sid_arg and sid_arg != "NEW":
        sid = sid_arg
    else:
        m = re.search(r"ses_[A-Za-z0-9]+", log)
        sid = m.group(0) if m else None

    out = clean(raw) or ("(the agent hit an error)" if status == "error" else "(no output)")
    db.finish(name, sid, status, out)


if __name__ == "__main__":
    main()

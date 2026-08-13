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
    return (f"Bet — I put an agent on '{name}'. It's grinding in the background; "
            f"ask me for its status whenever.")


def send_message(name: str, message: str) -> str:
    s = db.get(name)
    if not s:
        return f"I don't have a session called '{name}'. Want me to start one?"
    if not s.get("opencode_session_id"):
        return f"'{name}' is still spinning up — give it a sec, then try again."
    db.set_running(name, message)
    _launch(name, s["workdir"], s["opencode_session_id"], message)
    return f"Sent that into '{name}'. It's back to work."


def list_recent(limit: int = 6):
    return db.list_recent(limit)


def get_state(name: str):
    return db.get(name)


def delete_session(name: str) -> str:
    existed = db.delete(name)
    return (f"Deleted session '{name}' and its files."
            if existed else f"I don't have a session called '{name}'.")

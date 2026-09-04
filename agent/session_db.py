"""SQLite store for Kronik's named opencode agent sessions."""
import sqlite3, time, os, shutil

# Repo root derived from THIS file (dir above agent/), not a hardcoded ~/voice-agent, so a
# clone in any directory keeps its DB + working dirs beside the code, not in a phantom dir.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "sessions.db")
_WS = os.path.realpath(os.path.join(_ROOT, "sessions_ws"))


def _rm_workdir(workdir):
    """Safely remove a session's working dir (only if it's under sessions_ws)."""
    if not workdir:
        return
    try:
        real = os.path.realpath(workdir)
        if real.startswith(_WS + os.sep):
            shutil.rmtree(real, ignore_errors=True)
    except Exception:
        pass


def _conn():
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")   # allow agent reads while the runner writes
    return c


def init():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                name TEXT PRIMARY KEY,
                opencode_session_id TEXT,
                workdir TEXT,
                task TEXT,
                status TEXT,
                last_message TEXT,
                last_output TEXT,
                created_at REAL,
                updated_at REAL,
                announced_at REAL)"""
        )
    # Safe migration: add announced_at column if it doesn't exist (SQLite ignores errors
    # on ALTER TABLE when the column already exists, but be explicit).
    try:
        with _conn() as c:
            c.execute("ALTER TABLE sessions ADD COLUMN announced_at REAL")
    except Exception:
        pass  # column already exists — ignore


def create(name, task, workdir):
    """Register a new (or reset an existing) session, marked running."""
    init()
    now = time.time()
    with _conn() as c:
        c.execute(
            """INSERT INTO sessions(name,opencode_session_id,workdir,task,status,last_message,last_output,created_at,updated_at,announced_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                  task=excluded.task, status='running', last_message=excluded.task,
                  opencode_session_id=NULL, workdir=excluded.workdir, updated_at=excluded.updated_at,
                  announced_at=NULL""",
            (name, None, workdir, task, "running", task, "", now, now, None),
        )


def set_running(name, message):
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET status='running', last_message=?, updated_at=? WHERE name=? COLLATE NOCASE",
            (message, time.time(), name),
        )


def finish(name, session_id, status, output):
    """Called by the detached runner when an opencode run completes."""
    with _conn() as c:
        c.execute(
            """UPDATE sessions SET
                 opencode_session_id=COALESCE(?, opencode_session_id),
                 status=?, last_output=?, updated_at=?
               WHERE name=? COLLATE NOCASE""",
            (session_id, status, output, time.time(), name),
        )


def get(name):
    init()
    with _conn() as c:
        r = c.execute("SELECT * FROM sessions WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        return dict(r) if r else None


def mark_announced(name, completed_at=None):
    """Mark a session COMPLETION as announced. Sessions are multi-turn (message_session), so
    announcements are per-completion, not per-session: record the updated_at of the completion
    we actually announced — a LATER turn's completion bumps updated_at past this and gets its
    own announcement. (Recording the announced completion's own timestamp, not now(), closes
    the race where a new turn finishes while we're speaking the previous announcement.)"""
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET announced_at=? WHERE name=? COLLATE NOCASE",
            (completed_at if completed_at is not None else time.time(), name),
        )


def list_recent(limit=8):
    init()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete(name):
    """Delete a session (and its working dir) by name. Returns True if it existed."""
    init()
    with _conn() as c:
        r = c.execute(
            "SELECT workdir FROM sessions WHERE name=? COLLATE NOCASE", (name,)
        ).fetchone()
        if not r:
            return False
        c.execute("DELETE FROM sessions WHERE name=? COLLATE NOCASE", (name,))
    _rm_workdir(r["workdir"])
    return True


def prune_stale(ttl_days=30):
    """Delete sessions not updated in more than ttl_days (and their working dirs).
    'updated_at' is bumped on dispatch/resume/finish, so this removes sessions that
    haven't been chatted in recently. Returns the list of pruned session names."""
    init()
    cutoff = time.time() - ttl_days * 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT name, workdir FROM sessions WHERE updated_at < ?", (cutoff,)
        ).fetchall()
        pruned = [(r["name"], r["workdir"]) for r in rows]
        if pruned:
            c.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
    for _, wd in pruned:
        _rm_workdir(wd)
    return [n for n, _ in pruned]

"""TTL-based pruning: remove agent sessions not chatted in more than SESSION_TTL_DAYS
(default 30). Run periodically via launchd (local.kronik.prune). Logs what it removed."""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_db as db

TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "30"))

pruned = db.prune_stale(TTL_DAYS)
stamp = time.strftime("%Y-%m-%d %H:%M")
if pruned:
    print(f"{stamp} pruned {len(pruned)} stale session(s) (>{TTL_DAYS}d, no recent activity): {pruned}")
else:
    print(f"{stamp} no stale sessions to prune (TTL {TTL_DAYS}d)")

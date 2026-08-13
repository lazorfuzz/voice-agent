import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent"))
import agent_tools
print("DISPATCH:", agent_tools.dispatch("math-test", "What is 15 times 3? Answer in one short sentence."), flush=True)
for i in range(40):
    s = agent_tools.get_state("math-test")
    print(f"  [{i*4}s] status={s['status']} sid={s.get('opencode_session_id')}", flush=True)
    if s["status"] != "running": break
    time.sleep(4)
print("RESULT:", (s.get("last_output") or "")[:160], flush=True)
print("LIST:", [(r["name"], r["status"]) for r in agent_tools.list_recent()], flush=True)
print("RESUME:", agent_tools.send_message("math-test", "Now multiply that result by 2. One sentence."), flush=True)
for i in range(40):
    s = agent_tools.get_state("math-test")
    if s["status"] != "running": break
    time.sleep(4)
print("RESUMED RESULT:", (s.get("last_output") or "")[:160], flush=True)

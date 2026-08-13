#!/usr/bin/env python3
"""Unified onboarding CLI for the voice agent's home integrations.

    python agent/onboard.py <wiz|nest|tesla|ring|tv|petlibro|roomba>

Each flow LINKS the integration (auth / discovery / pairing), stores its secrets in
agent/.env.local, and flips its <NAME>_ENABLED flag so the voice + chat agent expose its
tools. Out of the box nothing is enabled except the OpenCode background-session tools; a
home integration only appears to the assistant after you onboard it here.
"""
import os
import sys
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_DIR, ".env.local"))
except Exception:
    pass

import config
import env_store

PY = sys.executable


def _run(script, *args):
    """Run one of the existing interactive auth scripts as a subprocess (inherits the tty so its
    email/password/PIN prompts work), returning its exit code."""
    return subprocess.call([PY, os.path.join(_DIR, script), *args])


# ---------- integrations whose existing auth script already persists its own secrets ----------
def onboard_ring():
    """Ring: interactive Amazon login (+ 2FA) -> RING_TOKEN_JSON."""
    if _run("ring_auth.py") == 0:
        env_store.enable("RING_ENABLED")
        print("\n✓ Ring enabled. Restart the agent to expose the camera tools.")


def onboard_petlibro():
    """PetLibro: interactive login -> email + password hash + token."""
    if _run("petlibro_auth.py") == 0:
        env_store.enable("PETLIBRO_ENABLED")
        print("\n✓ PetLibro enabled. Restart the agent to expose the feeder tools.")


def onboard_tesla():
    """Tesla Fleet API: needs a developer app + a public domain hosting the signing key first."""
    missing = [k for k in ("TESLA_CLIENT_ID", "TESLA_CLIENT_SECRET", "TESLA_REDIRECT_URI", "TESLA_DOMAIN")
               if not os.environ.get(k)]
    if missing:
        print("Tesla needs these in agent/.env.local first: " + ", ".join(missing))
        print("Create a Tesla developer app (developer.tesla.com), host its command-signing public key")
        print("at https://<your-domain>/.well-known/appspecific/com.tesla.3p.public-key.pem, then re-run.")
        return
    if _run("tesla_auth.py") == 0:
        env_store.enable("TESLA_ENABLED")
        print("\n✓ Tesla enabled (after you tap the virtual-key link on your phone). Restart the agent.")


# ---------- TV: PIN pairing, then remember the host ----------
def onboard_tv():
    """TV: pair over the Android TV Remote protocol (a PIN shows on the TV)."""
    host = input("TV IP address (blank to use the cached device): ").strip()
    if _run("tv_pair.py", *( [host] if host else [] )) == 0:
        if host:
            env_store.upsert_env("TV_HOST", host)
        env_store.enable("TV_ENABLED")
        print("\n✓ TV enabled. Restart the agent to expose the TV tools.")


# ---------- Roomba: fetch the LOCAL control password from the iRobot cloud ----------
def onboard_roomba():
    """Roomba: iRobot cloud login to fetch the local MQTT password + BLID, then local-only."""
    import getpass
    import roomba_cloud_pw as rc
    email = input("iRobot account email: ").strip()
    pw = getpass.getpass("iRobot password (hidden): ")
    rc_code = rc.main(email, pw)   # writes /tmp/roomba_blid.txt + /tmp/roomba_pw.txt on success
    pw = None
    if rc_code != 0:
        print("iRobot login failed.")
        return
    try:
        blid = open("/tmp/roomba_blid.txt").read().strip()
        rpw = open("/tmp/roomba_pw.txt").read().strip()
    except OSError:
        print("Couldn't read the fetched credentials.")
        return
    ip = input("Roomba's LAN IP address: ").strip()
    env_store.upsert_env("ROOMBA_BLID", blid)
    env_store.upsert_env("ROOMBA_PASSWORD", rpw)
    if ip:
        env_store.upsert_env("ROOMBA_IP", ip)
    for f in ("/tmp/roomba_blid.txt", "/tmp/roomba_pw.txt"):
        try:
            os.remove(f)
        except OSError:
            pass
    env_store.enable("ROOMBA_ENABLED")
    print("\n✓ Roomba enabled. Restart the agent to expose the vacuum tools.")


# ---------- WiZ lights: UDP-discover bulbs and name them ----------
def onboard_wiz():
    """WiZ: discover bulbs on the LAN over UDP, then give each a friendly name."""
    import time
    import wiz_tools
    print("Discovering WiZ bulbs on the LAN (UDP broadcast) ...")
    try:
        wiz_tools.prime_discovery()
        time.sleep(2)
    except Exception as e:
        print("  (discovery note:", e, ")")
    found = dict(getattr(wiz_tools, "_ip_by_mac", {}) or {})
    pairs = []
    if found:
        print(f"Found {len(found)} bulb(s). Give each a name you'll say out loud (e.g. 'bedside', 'kitchen').")
        for mac, ip in found.items():
            nm = input(f"  Name for bulb {mac} (at {ip}) [blank to skip]: ").strip()
            if nm:
                pairs.append(f"{nm}={mac}")
    else:
        print("No bulbs auto-discovered (they must be set up in the WiZ app on this LAN first).")
    while True:
        extra = input("  Add a bulb manually as name=mac (blank to finish): ").strip()
        if not extra:
            break
        pairs.append(extra)
    if not pairs:
        print("No bulbs configured — nothing enabled.")
        return
    env_store.upsert_env("WIZ_LIGHTS", ",".join(pairs))
    env_store.upsert_env("WIZ_BROADCAST", os.environ.get("WIZ_BROADCAST", "255.255.255.255"))
    env_store.enable("WIZ_ENABLED")
    print("\n✓ WiZ enabled:", ", ".join(pairs), "\nRestart the agent to expose the light tools.")


# ---------- Nest: guided (SDM needs a Google Cloud project set up first) ----------
def onboard_nest():
    """Nest: Google Smart Device Management. Requires external Google Cloud + OAuth setup."""
    print("Nest uses Google's Smart Device Management API, which needs a one-time external setup:")
    print("  1. Follow https://developers.google.com/nest/device-access/get-started")
    print("  2. Create a Device Access project (SDM), an OAuth client, and authorize your account")
    print("  3. Capture the OAuth refresh token and your thermostat's device id.")
    print("Then enter the values here:\n")
    fields = [("NEST_PROJECT_ID", "SDM project id"),
              ("NEST_CLIENT_ID", "OAuth client id"),
              ("NEST_CLIENT_SECRET", "OAuth client secret"),
              ("NEST_REFRESH_TOKEN", "OAuth refresh token"),
              ("NEST_DEVICE_ID", "thermostat device id")]
    vals = {}
    for key, prompt in fields:
        v = input(f"  {prompt}: ").strip()
        if not v:
            print("  (required — aborting, nothing saved)")
            return
        vals[key] = v
    for key, v in vals.items():
        env_store.upsert_env(key, v)
    env_store.enable("NEST_ENABLED")
    print("\n✓ Nest enabled. Restart the agent to expose the thermostat tools.")


# ---------- OpenCode background sessions (the run_agent/session tools) ----------
def onboard_opencode():
    """Verify opencode + a local-LLM provider, install the lean 'local' agent, enable session tools."""
    import shutil
    import json
    if not (os.path.exists(os.path.expanduser("~/.opencode/bin/opencode")) or shutil.which("opencode")):
        print("opencode is not installed. Install it, e.g.:")
        print("    curl -fsSL https://opencode.ai/install | bash")
        print("Then re-run:  python agent/onboard.py opencode")
        return
    cfgdir = os.path.expanduser("~/.config/opencode")
    os.makedirs(os.path.join(cfgdir, "agent"), exist_ok=True)
    # install the lean 'local' agent (the prompt that drives dispatched sessions)
    tmpl = os.path.join(_DIR, "templates", "opencode_local_agent.md")
    dest = os.path.join(cfgdir, "agent", "local.md")
    if os.path.exists(dest):
        print("opencode 'local' agent already present:", dest)
    elif os.path.exists(tmpl):
        shutil.copy(tmpl, dest)
        print("installed opencode 'local' agent ->", dest)
    # ensure a provider pointing at the local LLM
    ocjson = os.path.join(cfgdir, "opencode.json")
    if os.path.exists(ocjson):
        print("opencode.json already exists — leaving it (ensure it has a provider for your local LLM).")
    else:
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("LLM_MODEL", "").strip()
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"mlx-local": {
                "npm": "@ai-sdk/openai-compatible", "name": "Local LLM",
                "options": {"baseURL": base, "apiKey": key},
                "models": {(model or "local"): {"name": "Local model", "tools": True}}}},
            "default_agent": "local",
        }
        if model:
            cfg["model"] = f"mlx-local/{model}"
        with open(ocjson, "w") as f:
            json.dump(cfg, f, indent=2)
        print("wrote opencode provider config ->", ocjson)
    env_store.enable("OPENCODE_ENABLED")
    print("\n✓ OpenCode sessions enabled. Restart the agent to expose the run_agent/session tools.")


ONBOARDERS = {
    "opencode": onboard_opencode,
    "wiz": onboard_wiz, "nest": onboard_nest, "tesla": onboard_tesla, "ring": onboard_ring,
    "tv": onboard_tv, "petlibro": onboard_petlibro, "roomba": onboard_roomba,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ONBOARDERS:
        print("usage: python agent/onboard.py <" + "|".join(ONBOARDERS) + ">")
        print("\nCurrently enabled integrations:", ", ".join(sorted(config.enabled_integrations())) or "none")
        return 1
    ONBOARDERS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

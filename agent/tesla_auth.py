"""One-time Tesla Fleet API setup: register the partner domain, then OAuth login.

Run it yourself (your Tesla password stays in the browser, never here):

    cd ~/voice-agent && .venv/bin/python agent/tesla_auth.py

Steps it does:
  1. Partner registration — associates your domain (TESLA_DOMAIN, where the command-signing
     public key is hosted) with the app. One-time.
  2. Prints a Tesla login URL. Open it, log in, approve. Tesla redirects to your
     TESLA_REDIRECT_URI with ?code=... — copy that WHOLE url (or just the code) back here.
  3. Exchanges the code and saves ONLY the OAuth token to agent/.env.local (TESLA_TOKEN_JSON).

After this, tap the virtual-key link it prints (in the Tesla mobile app) to let the car
accept signed commands. tesla_tools.py refreshes the token from here on.
"""
import asyncio
import json
import os
import re
import time
from urllib.parse import urlparse, parse_qs

import aiohttp
from dotenv import load_dotenv

from tesla_fleet_api import TeslaFleetOAuth
from tesla_fleet_api.const import Scope

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))
_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")

CLIENT_ID = os.environ["TESLA_CLIENT_ID"]
CLIENT_SECRET = os.environ["TESLA_CLIENT_SECRET"]
REDIRECT_URI = os.environ["TESLA_REDIRECT_URI"]
DOMAIN = os.environ["TESLA_DOMAIN"]

SCOPES = [Scope.OPENID, Scope.OFFLINE_ACCESS, Scope.USER_DATA, Scope.VEHICLE_DEVICE_DATA,
          Scope.VEHICLE_LOCATION, Scope.VEHICLE_CMDS, Scope.VEHICLE_CHARGING_CMDS]


def _upsert_env(key: str, value: str):
    lines = open(_ENV).read().splitlines() if os.path.exists(_ENV) else []
    if any(l.startswith(key + "=") for l in lines):
        lines = [f"{key}={value}" if l.startswith(key + "=") else l for l in lines]
    else:
        lines.append(f"{key}={value}")
    open(_ENV, "w").write("\n".join(lines) + "\n")
    os.chmod(_ENV, 0o600)


async def main():
    async with aiohttp.ClientSession() as session:
        auth = TeslaFleetOAuth(session, region="na", client_id=CLIENT_ID,
                               client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI)

        print("1/3  Registering partner domain", DOMAIN, "…")
        try:
            pt = await auth.partner_login(CLIENT_ID, CLIENT_SECRET, [Scope.OPENID,
                    Scope.VEHICLE_DEVICE_DATA, Scope.VEHICLE_CMDS])
            # partner_login returns a client-credentials token but doesn't seat it; do so, else
            # register() sees expires=0, tries to refresh, and fails ("Refresh token is missing").
            auth._access_token = pt["access_token"]
            auth.expires = int(time.time()) + int(pt.get("expires_in", 28800))
            res = await auth.partner.register(DOMAIN)
            print("     ✓ registered:", json.dumps(res)[:120])
        except Exception as e:
            print("     (registration note:", str(e)[:160], ") — continuing; often already-registered is fine")

        url = auth.get_login_url(SCOPES, state="kronik")
        print("\n2/3  Open this URL, log in, and approve:\n")
        print("     " + url)
        print("\n     Then Tesla sends you to  " + REDIRECT_URI + "?code=…")
        pasted = input("\n     Paste that full URL (or just the code) here: ").strip()
        code = pasted
        if "code=" in pasted:
            code = parse_qs(urlparse(pasted).query).get("code", [pasted])[0]
        code = code.strip()

        print("\n3/3  Exchanging code for token …")
        await auth.get_refresh_token(code)
        token = {"access_token": auth._access_token, "refresh_token": auth.refresh_token,
                 "expires": auth.expires}
        _upsert_env("TESLA_TOKEN_JSON", json.dumps(token))
        print("     ✓ token saved to agent/.env.local (TESLA_TOKEN_JSON).")

    print("\n✓ Auth done. LAST STEP — authorize the car to accept commands:")
    print("  On your phone (with the Tesla app installed), open:")
    print("      https://tesla.com/_ak/" + DOMAIN)
    print("  It adds Kronik's key as a 'virtual key' to your car. Then tell me and I'll wire the tools.")


if __name__ == "__main__":
    asyncio.run(main())

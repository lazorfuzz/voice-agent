"""One-time interactive Ring login -> persists ONLY the OAuth refresh token to .env.local.

Run it yourself so your Amazon password stays in your terminal and is never seen by anyone
or written to disk:

    cd ~/voice-agent && .venv/bin/python agent/ring_auth.py

It prompts for your Ring/Amazon email, password (hidden), and the 2FA code, exchanges them
for an OAuth token, and writes RING_TOKEN_JSON=<json> into agent/.env.local. The password is
used once for the exchange and then discarded. ring_tools.py refreshes the token from here on.
"""
import asyncio
import getpass
import json
import os

from ring_doorbell import Auth
from ring_doorbell.exceptions import Requires2FAError

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
_UA = "kronik-voice-agent/1.0"


def _upsert_env(key: str, value: str):
    """Set KEY=value in .env.local (replace the line if present, else append)."""
    lines, found = [], False
    if os.path.exists(_ENV):
        with open(_ENV) as f:
            lines = f.read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(key + "="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    with open(_ENV, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(_ENV, 0o600)


async def main():
    user = input("Ring/Amazon email: ").strip()
    pw = getpass.getpass("Password (hidden): ")
    auth = Auth(_UA, None, None)
    try:
        token = await auth.async_fetch_token(user, pw)
    except Requires2FAError:
        otp = input("2FA code (texted/emailed to you): ").strip()
        token = await auth.async_fetch_token(user, pw, otp)
    finally:
        pw = None  # discard immediately
    _upsert_env("RING_TOKEN_JSON", json.dumps(token))
    await auth.async_close()
    print("\n✓ Ring token saved to agent/.env.local (RING_TOKEN_JSON). Password discarded.")
    print("  Next: I'll verify the connection and wire the camera tools into Kronik.")


if __name__ == "__main__":
    asyncio.run(main())

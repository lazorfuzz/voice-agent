"""One-time PetLibro login. Run it yourself so the password stays in your terminal:

    cd ~/voice-agent && .venv/bin/python agent/petlibro_auth.py

Stores your email, the password's MD5 hash (NOT the plaintext, which is discarded), and the
session token in agent/.env.local. Then prints your feeder(s) so we can wire "feed the cat".
"""
import asyncio
import getpass
import json
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))
import petlibro_tools as P


async def main():
    email = input("PetLibro email: ").strip()
    pw = getpass.getpass("PetLibro password (hidden): ")
    pmd5 = P.md5(pw)
    pw = None  # discard plaintext immediately

    P._upsert_env("PETLIBRO_EMAIL", email)
    P._upsert_env("PETLIBRO_PWD_MD5", pmd5)
    P._upsert_env("PETLIBRO_COUNTRY", os.environ.get("PETLIBRO_COUNTRY", "US"))
    # reload so petlibro_tools sees them
    os.environ["PETLIBRO_EMAIL"] = email
    os.environ["PETLIBRO_PWD_MD5"] = pmd5

    print("\nLogging in…")
    info = await P.probe()
    if P._session:
        await P._session.close()
    print("✓ Logged in. Token + password hash saved to agent/.env.local (plaintext discarded).\n")
    print("Feeders on your account:")
    for d in info["devices"]:
        print(f"  {d['name']!r}  product={d.get('product')}  sn={d['sn']}")
    print("\nLive data fields (so the status tool reports the right things):")
    print("  keys:", info.get("realInfo_keys"))
    print("  realInfo:", json.dumps(info.get("realInfo"), indent=2)[:900])


if __name__ == "__main__":
    asyncio.run(main())

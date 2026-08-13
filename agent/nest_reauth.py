"""Re-mint the Nest SDM OAuth refresh token (they die every 7 days while the Google Cloud
OAuth app is in "Testing" mode — publish it to production to stop the churn).

Run it YOURSELF (interactive Google login):  python agent/nest_reauth.py
1. Opens/prints the consent URL — approve in the browser.
2. Paste the redirected URL (or just the ?code=...) back here.
3. Exchanges it, verifies against the SDM API, and saves NEST_REFRESH_TOKEN to agent/.env.local.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))
from env_store import upsert_env

PROJECT = os.environ["NEST_PROJECT_ID"]
CLIENT = os.environ["NEST_CLIENT_ID"]
SECRET = os.environ["NEST_CLIENT_SECRET"]
# The SDM quickstart registers https://www.google.com as the redirect; override if yours differs.
REDIRECT = os.environ.get("NEST_REDIRECT_URI", "https://www.google.com")

url = ("https://nestservices.google.com/partnerconnections/"
       f"{PROJECT}/auth?" + urllib.parse.urlencode({
           "redirect_uri": REDIRECT,
           "access_type": "offline",
           "prompt": "consent",
           "client_id": CLIENT,
           "response_type": "code",
           "scope": "https://www.googleapis.com/auth/sdm.service",
       }))

print("\n1) Approve access in the browser (opening now):\n\n" + url + "\n")
try:
    webbrowser.open(url)
except Exception:
    pass

pasted = input("2) Paste the FULL redirected URL (or just the code): ").strip()
if "code=" in pasted:
    code = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)["code"][0]
else:
    code = pasted

body = urllib.parse.urlencode({
    "client_id": CLIENT, "client_secret": SECRET, "code": code,
    "grant_type": "authorization_code", "redirect_uri": REDIRECT,
}).encode()
tok = json.load(urllib.request.urlopen("https://oauth2.googleapis.com/token", data=body, timeout=15))
refresh = tok["refresh_token"]
access = tok["access_token"]

# sanity check against the live API before persisting
req = urllib.request.Request(
    f"https://smartdevicemanagement.googleapis.com/v1/enterprises/{PROJECT}/devices",
    headers={"Authorization": f"Bearer {access}"})
devices = json.load(urllib.request.urlopen(req, timeout=15)).get("devices", [])
print(f"\nSDM API OK — {len(devices)} device(s) visible.")

upsert_env("NEST_REFRESH_TOKEN", refresh)
print("Saved NEST_REFRESH_TOKEN to agent/.env.local.")
print("\nNOTE: restart the voice worker to pick it up, and to stop the 7-day expiry for good:")
print("  Google Cloud Console -> APIs & Services -> OAuth consent screen -> PUBLISH APP")

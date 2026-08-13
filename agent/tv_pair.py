#!/usr/bin/env python3
"""One-time pairing for Kronik's Android TV Remote control.

Run this ONCE. The TV will show a 6-character PIN; type it in when prompted.
It saves a client cert/key to ~/voice-agent so Kronik can drive the TV
(open apps, search, and actually play a title) from then on.

    ~/voice-agent/.venv/bin/python ~/voice-agent/agent/tv_pair.py [HOST]

HOST defaults to the TV cached in tv_device.json (or the TV_HOST env var).
"""
import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, not ~/voice-agent
CERT = os.path.join(_ROOT, "tv_remote_cert.pem")
KEY = os.path.join(_ROOT, "tv_remote_key.pem")
_CACHE = os.path.join(_ROOT, "tv_device.json")


def _host_from_arg_or_cache() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    try:
        return json.load(open(_CACHE))["host"]
    except Exception:
        return os.environ.get("TV_HOST", "")


async def main() -> int:
    from androidtvremote2 import AndroidTVRemote

    host = _host_from_arg_or_cache()
    print(f"Pairing with the TV at {host} ...")
    remote = AndroidTVRemote("Kronik", CERT, KEY, host)
    await remote.async_generate_cert_if_missing()

    try:
        name, mac = await remote.async_get_name_and_mac()
        print(f"Found: {name} ({mac})")
    except Exception:
        pass  # not fatal; some TVs answer this only post-pair

    await remote.async_start_pairing()
    print("\n>>> A 4-6 character PIN should now be showing ON THE TV SCREEN. <<<", flush=True)

    pin_file = os.path.join(_ROOT, "tv_pin.txt")
    try:
        os.remove(pin_file)
    except OSError:
        pass

    if sys.stdin and sys.stdin.isatty():
        pin = input("Type the PIN here and press Enter: ").strip()
    else:
        # Non-interactive: wait for the PIN to be dropped into tv_pin.txt.
        print(f"Waiting for the PIN in {pin_file} (up to 180s) ...", flush=True)
        pin = ""
        for _ in range(360):
            if os.path.exists(pin_file):
                pin = open(pin_file).read().strip()
                if pin:
                    break
            await asyncio.sleep(0.5)
        try:
            os.remove(pin_file)
        except OSError:
            pass
        if not pin:
            print("No PIN received in time.")
            return 2
    print(f"Using PIN: {pin}", flush=True)

    await remote.async_finish_pairing(pin)
    print("\n✅ Paired! Verifying the connection ...")

    # confirm the saved cert actually works
    await remote.async_connect()
    print(f"Connected. Current app on TV: {remote.current_app!r}")
    remote.disconnect()
    print(f"Saved credentials to:\n  {CERT}\n  {KEY}\nKronik can now control the TV.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"\n❌ Pairing failed ({type(e).__name__}): {e}")
        print("Make sure the TV is ON and on the same network, then try again.")
        raise SystemExit(1)

#!/usr/bin/env python
"""Emit a *runtime* LiveKit config that only binds/advertises addresses that currently
exist on a network interface.

LiveKit binds EVERY entry in `bind_addresses` (and advertises every `rtc.ips.includes`)
or it refuses to start at all. So a tunnel/VPN address that isn't up yet — e.g. a WireGuard
IP after a reboot — takes the whole media server down, even for localhost. run_all.sh calls
this before starting LiveKit so a downed tunnel can't wedge local voice: absent addresses
are dropped, loopback is always kept, and remote self-heals on the next run once the tunnel
returns.

Usage: python livekit_runtime_config.py <src.yaml> <dst.yaml>
Exit 0 on success (dst written). On any error, exits non-zero so the caller can fall back
to the source config unchanged.
"""
import subprocess
import sys

import yaml

LOOPBACK = {"127.0.0.1", "::1", "0.0.0.0", "::"}


def present_addrs() -> set:
    """IPv4/IPv6 addresses currently assigned to a local interface (macOS `ifconfig`)."""
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    addrs = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("inet ") or line.startswith("inet6 "):
            parts = line.split()
            if len(parts) >= 2:
                addrs.add(parts[1].split("%")[0])  # strip zone id on inet6
    return addrs


def keep_addr(addr: str, present: set) -> bool:
    return addr in LOOPBACK or addr in present


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}

    present = present_addrs()
    dropped = []

    # --- bind_addresses: keep loopback + present; never let it go empty ---
    binds = cfg.get("bind_addresses")
    if isinstance(binds, list):
        kept = [a for a in binds if keep_addr(str(a), present)]
        dropped += [f"bind:{a}" for a in binds if not keep_addr(str(a), present)]
        if not kept:
            kept = ["127.0.0.1"]
        cfg["bind_addresses"] = kept

    # --- rtc.ips.includes: keep present (strip /mask before the check) ---
    rtc = cfg.get("rtc")
    if isinstance(rtc, dict):
        ips = rtc.get("ips")
        if isinstance(ips, dict) and isinstance(ips.get("includes"), list):
            inc = ips["includes"]
            kept = [e for e in inc if keep_addr(str(e).split("/")[0], present)]
            dropped += [f"rtc.ips:{e}" for e in inc if not keep_addr(str(e).split("/")[0], present)]
            if kept:
                ips["includes"] = kept
            else:
                # No advertised IP left — drop the block so LiveKit falls back to node_ip.
                rtc.pop("ips", None)

    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    if dropped:
        print("  [livekit] dropped absent address(es): " + ", ".join(dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

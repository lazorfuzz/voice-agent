"""Persist secrets/settings into agent/.env.local — the one place onboarding writes to.

Used by the onboarding CLI (onboard.py) to save tokens and flip integration enable flags.
Existing per-integration auth scripts have their own copies of this for historical reasons;
new code should use this module.
"""
import os

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")


def upsert_env(key: str, value: str, path: str = None):
    """Set KEY=value in .env.local (replace the line if present, else append). chmod 600, and
    reflect it into the current process env so a subsequent step in the same run sees it."""
    p = path or _ENV
    lines = open(p).read().splitlines() if os.path.exists(p) else []
    newline = f"{key}={value}"
    for i, ln in enumerate(lines):
        if ln.startswith(key + "="):
            lines[i] = newline
            break
    else:
        lines.append(newline)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    os.environ[key] = value


def enable(flag: str):
    """Turn an integration on by setting its <NAME>_ENABLED flag to true."""
    upsert_env(flag, "true")

"""Background bridge from the Mage/Ring probe to spoken LiveKit commentary.

Mage deliberately runs in its isolated virtual environment and process. This
keeps its ~11 GB checkpoint, PyAV version, and MPS work out of the latency-
sensitive voice worker. The probe writes JSONL; this bridge speaks only new
descriptions and drops commentary that has gone stale while the user is busy.
"""
from __future__ import annotations

import asyncio
import contextlib
from difflib import SequenceMatcher
import json
import logging
import os
from pathlib import Path
import re
from typing import Any


LOG = logging.getLogger("kronik.mage_ring")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".mage-venv" / "bin" / "python"
PROBE = ROOT / "experimental" / "mage_ring_live.py"


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).split())


def is_repeat(current: str, previous: str | None, threshold: float = 0.84) -> bool:
    """Return whether two short camera descriptions are effectively identical."""
    if not previous:
        return False
    current_norm = _normalized(current)
    previous_norm = _normalized(previous)
    if not current_norm or not previous_norm:
        return current_norm == previous_norm
    return (
        current_norm == previous_norm
        or SequenceMatcher(None, current_norm, previous_norm).ratio() >= threshold
    )


class MageRingWatcher:
    """Own one on-demand Mage subprocess for the lifetime of a voice session."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._camera = ""
        self._last_spoken: str | None = None
        self._has_spoken = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, session: Any, camera: str, minutes: float) -> tuple[bool, str]:
        if self.running:
            return False, f"I am already watching the {self._camera or 'Ring'} camera."
        python = Path(os.environ.get("MAGE_RING_PYTHON", str(DEFAULT_PYTHON)))
        if not python.is_file():
            return False, (
                "Mage's isolated environment is not installed. Run the setup in "
                "experimental/MAGE_RING.md first."
            )
        if not PROBE.is_file():
            return False, "The Mage Ring probe is missing."

        self._camera = (camera or "bedroom").strip()
        self._last_spoken = None
        self._has_spoken = False
        duration = max(0.5, min(float(minutes), 10.0)) * 60.0
        self._task = asyncio.create_task(
            self._run(session, self._camera, duration),
            name=f"mage-ring-{self._camera}",
        )
        return True, (
            f"Started watching the {self._camera} camera with Mage for "
            f"{duration / 60:g} minute(s). The first update can take about 15 seconds."
        )

    async def stop(self) -> bool:
        task = self._task
        process = self._process
        if task is None or task.done():
            self._task = None
            return False
        if process is not None and process.returncode is None:
            process.terminate()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._process = None
        return True

    async def _wait_until_quiet(self, session: Any, max_wait: float = 12.0) -> bool:
        """Avoid talking over the user/agent; discard an update once it is stale."""
        deadline = asyncio.get_running_loop().time() + max_wait
        while asyncio.get_running_loop().time() < deadline:
            # LiveKit changes an inactive user's state from "listening" to
            # "away" after a silence timeout. Away is exactly when unsolicited
            # camera commentary should play; only active speech is busy.
            if (
                getattr(session, "user_state", "listening") != "speaking"
                and getattr(session, "agent_state", "listening") == "listening"
            ):
                return True
            await asyncio.sleep(0.25)
        return False

    async def _drain_stderr(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while line := await stream.readline():
            LOG.debug("Mage: %s", line.decode(errors="replace").rstrip())

    async def _run(self, session: Any, camera: str, duration: float) -> None:
        python = Path(os.environ.get("MAGE_RING_PYTHON", str(DEFAULT_PYTHON)))
        args = [
            str(python), str(PROBE),
            "--camera", camera,
            "--skip-gate",
            "--segment-seconds", os.environ.get("MAGE_RING_SEGMENT_SECONDS", "8"),
            "--sample-fps", os.environ.get("MAGE_RING_SAMPLE_FPS", "2"),
            "--max-pixels", os.environ.get("MAGE_RING_MAX_PIXELS", "150000"),
            "--max-new-tokens", os.environ.get("MAGE_RING_MAX_TOKENS", "32"),
            "--run-seconds", f"{duration:g}",
            # Active duration is owned by --run-seconds; this is only a generous
            # safety ceiling for the number of windows.
            "--max-segments", "100000",
        ]
        stderr_task: asyncio.Task | None = None
        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"},
            )
            stderr_task = asyncio.create_task(self._drain_stderr(self._process.stderr))
            await asyncio.wait_for(
                self._consume(session, self._process.stdout),
                # The requested duration begins when Ring media is ready, not
                # while Mage loads and the camera wakes. Keep a bounded startup
                # allowance around the subprocess's own --run-seconds timer.
                timeout=duration + 45.0,
            )
        except TimeoutError:
            LOG.info("Mage watch duration elapsed for %s", camera)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Mage camera watch failed for %s", camera)
        finally:
            process = self._process
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if stderr_task is not None:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
            self._process = None

    async def _consume(
        self,
        session: Any,
        stream: asyncio.StreamReader | None,
    ) -> None:
        if stream is None:
            raise RuntimeError("Mage subprocess stdout was not connected")
        while line := await stream.readline():
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                LOG.debug("Ignoring non-JSON Mage output: %r", line[:200])
                continue
            if event.get("type") == "error":
                LOG.error("Mage camera probe error: %s", event.get("error"))
                if await self._wait_until_quiet(session, max_wait=3.0):
                    await session.say(
                        "I couldn't continue the live camera watch.",
                        allow_interruptions=True,
                        add_to_chat_ctx=False,
                    )
                continue
            if event.get("type") != "commentary" or event.get("decision") != "response":
                continue
            text = str(event.get("text") or "").strip().rstrip(".")
            if not text or is_repeat(text, self._last_spoken):
                continue
            LOG.info(
                "Mage commentary ready: %s (user=%s agent=%s)",
                text,
                getattr(session, "user_state", "?"),
                getattr(session, "agent_state", "?"),
            )
            if not await self._wait_until_quiet(session):
                LOG.info(
                    "Dropped stale Mage commentary while voice session was busy: %s "
                    "(user=%s agent=%s)",
                    text,
                    getattr(session, "user_state", "?"),
                    getattr(session, "agent_state", "?"),
                )
                continue
            self._last_spoken = text
            spoken_text = (
                f"{self._camera} camera: {text}." if not self._has_spoken
                else f"{text}."
            )
            self._has_spoken = True
            await session.say(
                spoken_text,
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )

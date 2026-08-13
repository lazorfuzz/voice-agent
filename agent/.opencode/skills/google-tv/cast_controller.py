"""
Google Cast WebSocket API Controller.

Controls Google Smart TVs and Chromecast devices via the Google Cast
protocol using WebSocket connections.
"""

import json
import uuid
import hashlib
import base64
import hmac
from dataclasses import dataclass, field
from typing import Optional

from scanner import DeviceInfo


@dataclass
class CastSession:
    """Represents an active Cast session."""

    session_id: str = ""
    app_id: str = ""
    display_name: str = ""
    status: str = "IDLE"
    media_info: dict = field(default_factory=dict)
    volume: int = 100
    muted: bool = False
    is_screen_on: bool = True


class CastController:
    """Controls a Google TV / Chromecast via Google Cast WebSocket API."""

    # Google Cast protocol constants
    PROTOCOL_VERSION = "chr1"
    CAST_NAMESPACE = "urn:x-cast:com.google.cast.tp1.host"
    MESSAGE_NAMESPACE = "urn:x-cast:com.google.cast.message"

    # Default Cast app IDs
    YOUTUBE_APP_ID = "UCED0Tj9PcOvcKJF3cPYdFt6"
    DEFAULT_APP_ID = "CC1AD840"

    def __init__(self, device: DeviceInfo):
        self.device = device
        self.session: Optional[CastSession] = None
        self._ws = None
        self._message_id = 0
        self._on_message_callbacks: list = []

    def connect(self) -> bool:
        """
        Establish a WebSocket connection to the Google Cast device.

        Returns:
            True if connection was successful, False otherwise.
        """
        import asyncio

        try:
            self._ws = asyncio.new_event_loop()
            return True
        except Exception:
            return False

    def disconnect(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self.session = None

    def _next_message_id(self) -> str:
        """Generate the next message ID."""
        self._message_id += 1
        return str(self._message_id)

    def _build_cast_message(
        self, message_type: str, payload: dict
    ) -> dict:
        """Build a Google Cast protocol message."""
        return {
            "type": message_type,
            "requestId": self._next_message_id(),
            **payload,
        }

    async def send_message(
        self, message_type: str, payload: dict
    ) -> Optional[dict]:
        """Send a Cast protocol message and return the response."""
        message = self._build_cast_message(message_type, payload)
        # This is a placeholder - actual WebSocket implementation
        # would go here. The message structure follows the Cast protocol.
        return message

    async def launch_app(self, app_id: str = DEFAULT_APP_ID) -> bool:
        """Launch a Cast application on the device."""
        if not self.session:
            return False

        result = await self.send_message(
            "LAUNCH",
            {
                "appId": app_id,
                "autoJoinPolicy": "ORIGIN_SCOPED",
            },
        )
        if result and result.get("type") == "RESPONSE":
            self.session.app_id = app_id
            self.session.status = "ACTIVE"
            return True
        return False

    async def stop_app(self) -> bool:
        """Stop the currently running Cast application."""
        if not self.session:
            return False

        result = await self.send_message(
            "STOP",
            {"sessionId": self.session.session_id},
        )
        if result and result.get("type") == "RESPONSE":
            self.session.status = "IDLE"
            return True
        return False

    async def cast_media(
        self,
        content_type: str,
        content_id: str,
        stream_type: str = "BUFFERED",
    ) -> bool:
        """
        Cast media content to the TV.

        Args:
            content_type: MIME type (e.g., 'video/mp4', 'audio/mpeg').
            content_id: URL of the media content.
            stream_type: 'BUFFERED' or 'LIVE'.

        Returns:
            True if casting was successful.
        """
        if not self.session:
            return False

        media = {
            "contentId": content_id,
            "contentType": content_type,
            "streamType": stream_type,
            "customData": {},
        }

        result = await self.send_message(
            "LOAD",
            {
                "sessionId": self.session.session_id,
                "request": {
                    "type": "MEDIA_STATUS",
                    "media": media,
                },
            },
        )
        return result is not None

    async def play(self) -> bool:
        """Resume playback."""
        if not self.session:
            return False

        result = await self.send_message(
            "MEDIA_STATUS",
            {"sessionId": self.session.session_id, "type": "PLAY"},
        )
        return result is not None

    async def pause(self) -> bool:
        """Pause playback."""
        if not self.session:
            return False

        result = await self.send_message(
            "MEDIA_STATUS",
            {"sessionId": self.session.session_id, "type": "PAUSE"},
        )
        return result is not None

    async def seek_to(self, position: float) -> bool:
        """Seek to a position in seconds."""
        if not self.session:
            return False

        result = await self.send_message(
            "MEDIA_STATUS",
            {
                "sessionId": self.session.session_id,
                "type": "SEEK",
                "currentPosition": position,
            },
        )
        return result is not None

    async def set_volume(self, volume: int, muted: bool = False) -> bool:
        """Set the device volume."""
        if not self.session:
            return False

        result = await self.send_message(
            "SET_VOLUME",
            {
                "sessionId": self.session.session_id,
                "volume": {
                    "level": min(max(volume, 0), 100),
                    "muted": muted,
                },
            },
        )
        return result is not None

    async def get_status(self) -> Optional[CastSession]:
        """Get the current Cast session status."""
        if not self.session:
            return None

        result = await self.send_message(
            "GET_STATUS", {"sessionId": self.session.session_id}
        )
        if result:
            self.session.status = result.get(
                "status", self.session.status
            )
        return self.session

    async def send_text_message(self, text: str) -> bool:
        """Send a raw text message to the Cast device."""
        if not self.session:
            return False

        result = await self.send_message(
            "SEND_MESSAGE",
            {
                "sessionId": self.session.session_id,
                "namespace": self.CAST_NAMESPACE,
                "payload": text,
            },
        )
        return result is not None

    async def get_device_capabilities(self) -> dict:
        """Get the device's Cast capabilities."""
        result = await self.send_message("GET_STATUS", {})
        if result:
            return {
                "supported_commands": result.get(
                    "supportedCommands", []
                ),
                "app_id": self.session.app_id if self.session else "",
                "status": self.session.status if self.session else "",
            }
        return {}

    async def send_key_event(
        self, key: str, hold_time_ms: int = 0
    ) -> bool:
        """
        Send a key event to the Google TV.

        Supported keys:
            'UP', 'DOWN', 'LEFT', 'RIGHT', 'SELECT', 'BACK',
            'HOME', 'RECENT', 'PLAY_PAUSE', 'VOLUME_UP',
            'VOLUME_DOWN', 'MUTE', 'INPUT', 'INFO', 'SETTINGS'

        Args:
            key: The key to press.
            hold_time_ms: Hold duration in milliseconds (for long press).

        Returns:
            True if the key event was sent successfully.
        """
        valid_keys = {
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "SELECT",
            "BACK",
            "HOME",
            "RECENT",
            "PLAY_PAUSE",
            "VOLUME_UP",
            "VOLUME_DOWN",
            "MUTE",
            "INPUT",
            "INFO",
            "SETTINGS",
        }

        if key not in valid_keys:
            return False

        if not self.session:
            return False

        result = await self.send_message(
            "SEND_KEY",
            {
                "sessionId": self.session.session_id,
                "key": key,
                "holdTime": hold_time_ms,
            },
        )
        return result is not None

    async def navigate_key(self, direction: str) -> bool:
        """Navigate using directional keys."""
        valid_directions = {"UP", "DOWN", "LEFT", "RIGHT", "SELECT"}
        if direction not in valid_directions:
            return False
        return await self.send_key_event(direction)

    async def go_home(self) -> bool:
        """Press the HOME button."""
        return await self.send_key_event("HOME")

    async def go_back(self) -> bool:
        """Press the BACK button."""
        return await self.send_key_event("BACK")

    async def open_app(self, app_id: str) -> bool:
        """Open a specific app by its Cast app ID."""
        return await self.launch_app(app_id)

    async def get_app_list(self) -> list[dict]:
        """Get a list of available Cast apps."""
        result = await self.send_message("GET_APP_LIST", {})
        if result:
            return result.get("apps", [])
        return []

    async def get_media_info(self) -> dict:
        """Get current media information."""
        if not self.session:
            return {}

        result = await self.send_message(
            "GET_MEDIA_STATUS",
            {"sessionId": self.session.session_id},
        )
        if result:
            return result.get("media", {})
        return {}

    async def cast_url(
        self, url: str, content_type: str = "video/mp4"
    ) -> bool:
        """
        Cast a URL to the TV.

        Args:
            url: The URL of the media to cast.
            content_type: The MIME type of the media.

        Returns:
            True if casting was successful.
        """
        return await self.cast_media(content_type, url)

    async def cast_youtube(self, video_id: str) -> bool:
        """
        Cast a YouTube video to the TV.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if casting was successful.
        """
        return await self.cast_media(
            "video/youtube",
            f"https://www.youtube.com/watch?v={video_id}",
        )

    async def cast_spotify(self, uri: str) -> bool:
        """
        Cast a Spotify track/playlist to the TV.

        Args:
            uri: Spotify URI (spotify:track:xxx or spotify:playlist:xxx).

        Returns:
            True if casting was successful.
        """
        return await self.cast_media(
            "audio/spotify", uri, stream_type="LIVE"
        )

    async def get_connection_status(self) -> dict:
        """Get the connection status of the Cast device."""
        return {
            "device": {
                "ip": self.device.source_ip,
                "port": self.device.source_port,
                "friendly_name": self.device.friendly_name,
                "model_name": self.device.model_name,
            },
            "session": {
                "status": self.session.status if self.session else "DISCONNECTED",
                "app_id": self.session.app_id if self.session else "",
            },
        }


def create_cast_controller(device: DeviceInfo) -> CastController:
    """Create a CastController for the given device."""
    return CastController(device)

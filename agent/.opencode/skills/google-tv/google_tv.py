"""
Google TV Skill - Main Controller.

A unified interface for discovering and controlling Google Smart TVs
and Chromecast devices on the local network.

Supports:
  - SSDP network discovery
  - UPnP/DLNA control (volume, playback, inputs)
  - Google Cast WebSocket API (key presses, media casting)
  - Automatic fallback between control methods
"""

from scanner import (
    DeviceInfo,
    discover_google_tvs,
    discover_all_devices,
    print_device_info,
)
from upnp_controller import UPnPController, create_upnp_controller
from cast_controller import CastController, create_cast_controller


class GoogleTV:
    """
    Unified controller for Google Smart TVs and Chromecast devices.

    Automatically discovers devices on the network and provides
    multiple control methods (UPnP/DLNA and Google Cast).
    """

    def __init__(self, device: DeviceInfo):
        self.device = device
        self.upnp: UPnPController = create_upnp_controller(device)
        self.cast: CastController = create_cast_controller(device)
        self._preferred_method = self._determine_preferred_method()

    def _determine_preferred_method(self) -> str:
        """Determine the best control method for this device."""
        if self.device.has_cast_api:
            return "cast"
        if self.upnp.av_transport_url:
            return "upnp"
        return "none"

    @property
    def is_online(self) -> bool:
        """Check if the device is reachable."""
        return self._preferred_method != "none"

    @property
    def friendly_name(self) -> str:
        """Get the device's friendly name."""
        return self.device.friendly_name

    @property
    def model_name(self) -> str:
        """Get the device model name."""
        return self.device.model_name

    @property
    def is_google_tv(self) -> bool:
        """Check if this is a Google TV (not just Chromecast)."""
        return self.device.is_google_tv

    @property
    def is_chromecast(self) -> bool:
        """Check if this is a Chromecast device."""
        return self.device.is_chromecast

    def get_status(self) -> dict:
        """Get comprehensive device status."""
        status = {
            "device": {
                "friendly_name": self.device.friendly_name,
                "model_name": self.device.model_name,
                "manufacturer": self.device.manufacturer,
                "uuid": self.device.uuid,
                "is_google_tv": self.device.is_google_tv,
                "is_chromecast": self.device.is_chromecast,
                "source_ip": self.device.source_ip,
                "preferred_method": self._preferred_method,
            },
            "upnp": {
                "available": bool(self.upnp.av_transport_url),
                "services": self.device.services,
            },
            "cast": {
                "available": self.device.has_cast_api,
                "connection": {"status": "sync_unavailable"},
            },
        }
        return status

    # --- Volume Control ---

    def get_volume(self) -> int:
        """Get current volume level (0-100)."""
        return self.upnp.get_device_volume().volume

    def set_volume(self, volume: int) -> bool:
        """Set volume level (0-100)."""
        return self.upnp.set_volume(volume)

    def volume_up(self) -> bool:
        """Increase volume by 1."""
        return self.upnp.volume_up()

    def volume_down(self) -> bool:
        """Decrease volume by 1."""
        return self.upnp.volume_down()

    def set_mute(self, muted: bool) -> bool:
        """Mute or unmute the TV."""
        return self.upnp.set_mute(muted)

    def get_mute_state(self) -> bool:
        """Get current mute state."""
        return self.upnp.get_device_volume().mute

    # --- Playback Control ---

    def get_transport_state(self) -> str:
        """Get current transport state (STOPPED, PLAYING, PAUSED, etc.)."""
        return self.upnp.get_transport_state().current_state

    def play(self) -> bool:
        """Start/resume playback."""
        return self.upnp.play()

    def pause(self) -> bool:
        """Pause playback."""
        return self.upnp.pause()

    def stop(self) -> bool:
        """Stop playback."""
        return self.upnp.stop()

    def seek(self, track_number: int = 0, track_time: str = "") -> bool:
        """Seek to a specific position."""
        return self.upnp.seek(track_number, track_time)

    def get_position_info(self) -> dict:
        """Get current playback position."""
        return self.upnp.get_position_info()

    # --- Input/Source Control ---

    def get_inputs(self) -> list[dict]:
        """Get available input sources."""
        return self.upnp.get_available_inputs()

    def set_input(self, uri: str, metadata: str = "") -> bool:
        """Switch to a specific input source."""
        return self.upnp.set_transport_url(uri, metadata)

    # --- Navigation Keys ---

    async def press_key(self, key: str, hold_time_ms: int = 0) -> bool:
        """
        Press a navigation key on the Google TV.

        Supported keys:
            'UP', 'DOWN', 'LEFT', 'RIGHT', 'SELECT', 'BACK',
            'HOME', 'RECENT', 'PLAY_PAUSE', 'VOLUME_UP',
            'VOLUME_DOWN', 'MUTE', 'INPUT', 'INFO', 'SETTINGS'

        Args:
            key: The key to press.
            hold_time_ms: Hold duration in milliseconds.

        Returns:
            True if the key event was sent successfully.
        """
        return await self.cast.send_key_event(key, hold_time_ms)

    async def navigate(self, direction: str) -> bool:
        """Navigate using directional keys."""
        return await self.cast.navigate_key(direction)

    async def go_home(self) -> bool:
        """Press the HOME button."""
        return await self.cast.go_home()

    async def go_back(self) -> bool:
        """Press the BACK button."""
        return await self.cast.go_back()

    # --- Media Casting ---

    async def cast_url(self, url: str, content_type: str = "video/mp4") -> bool:
        """Cast a media URL to the TV."""
        return await self.cast.cast_url(url, content_type)

    async def cast_youtube(self, video_id: str) -> bool:
        """Cast a YouTube video to the TV."""
        return await self.cast.cast_youtube(video_id)

    async def cast_spotify(self, uri: str) -> bool:
        """Cast a Spotify track/playlist to the TV."""
        return await self.cast.cast_spotify(uri)

    async def launch_app(self, app_id: str) -> bool:
        """Launch a Cast application."""
        return await self.cast.launch_app(app_id)

    async def stop_casting(self) -> bool:
        """Stop the currently running Cast application."""
        return await self.cast.stop_app()

    # --- Device Info ---

    def get_device_info(self) -> dict:
        """Get detailed device information."""
        return self.upnp.get_device_info()

    def get_transport_info(self) -> dict:
        """Get current transport/media information."""
        return self.upnp.get_transport_info()

    def __repr__(self) -> str:
        return (
            f"GoogleTV(name='{self.device.friendly_name}', "
            f"model='{self.device.model_name}', "
            f"method='{self._preferred_method}', "
            f"ip='{self.device.source_ip}')"
        )


def discover_and_control() -> list[GoogleTV]:
    """
    Discover all Google TVs and Chromecast devices on the network.

    Returns:
        List of GoogleTV controller instances for discovered devices.
    """
    devices = discover_google_tvs()
    return [GoogleTV(device) for device in devices]


def discover_all() -> list[GoogleTV]:
    """
    Discover all UPnP devices and filter for Google TVs.

    Returns:
        List of GoogleTV controller instances.
    """
    all_devices = discover_all_devices()
    return [GoogleTV(device) for device in all_devices if device.is_google_tv]


def print_tv_info(tv: GoogleTV) -> str:
    """Print formatted information about a Google TV."""
    lines = [
        f"GoogleTV: {tv.friendly_name}",
        f"  Model: {tv.model_name}",
        f"  IP: {tv.device.source_ip}",
        f"  Method: {tv._preferred_method}",
        f"  Online: {tv.is_online}",
        f"  Google TV: {tv.is_google_tv}",
        f"  Chromecast: {tv.is_chromecast}",
    ]
    return "\n".join(lines)

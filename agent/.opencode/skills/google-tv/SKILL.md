# Google TV Skill

Discover and control Google Smart TVs and Chromecast devices on your local network.

## Capabilities

- **Network Discovery**: Find Google TVs and Chromecast devices via SSDP multicast
- **UPnP/DLNA Control**: Volume, playback, input switching via UPnP protocols
- **Google Cast API**: Key presses, media casting, app management via WebSocket
- **Automatic Fallback**: Chooses the best control method for each device

## Quick Start

```python
from google_tv import discover_and_control, GoogleTV

# Discover all Google TVs on the network
tvs = discover_and_control()

for tv in tvs:
    print(tv)
    # GoogleTV(name='Living Room TV', model='XR65A80J', method='upnp', ip='10.0.0.113')

# Control a specific TV
tv = tvs[0]

# Volume control
tv.set_volume(50)
tv.volume_up()
tv.volume_down()
tv.set_mute(True)
current_vol = tv.get_volume()

# Playback control
tv.play()
tv.pause()
tv.stop()
tv.seek(track_time="00:05:30")

# Navigation (requires Cast API)
import asyncio
asyncio.run(tv.go_home())
asyncio.run(tv.go_back())
asyncio.run(tv.navigate("UP"))
asyncio.run(tv.navigate("SELECT"))

# Media casting
asyncio.run(tv.cast_url("https://example.com/video.mp4", "video/mp4"))
asyncio.run(tv.cast_youtube("dQw4w9WgXcQ"))
```

## API Reference

### Discovery

| Function | Description |
|---|---|
| `discover_and_control()` | Find all Google TVs/Chromecasts, return `GoogleTV` instances |
| `discover_all()` | Same as above, includes all UPnP devices |
| `print_device_info(device)` | Print formatted device info string |

### GoogleTV Controller

| Property/Method | Description |
|---|---|
| `tv.friendly_name` | Device name (e.g., "Living Room TV") |
| `tv.model_name` | Model number (e.g., "XR65A80J") |
| `tv.is_google_tv` | Boolean: is this a Google TV? |
| `tv.is_chromecast` | Boolean: is this a Chromecast? |
| `tv.is_online` | Boolean: is the device reachable? |
| `tv.get_status()` | Full device status dict |
| `tv.get_volume()` | Current volume (0-100) |
| `tv.set_volume(n)` | Set volume (0-100) |
| `tv.volume_up()` | Increase volume by 1 |
| `tv.volume_down()` | Decrease volume by 1 |
| `tv.set_mute(bool)` | Mute/unmute |
| `tv.get_mute_state()` | Current mute state |
| `tv.get_transport_state()` | Current state: STOPPED, PLAYING, PAUSED |
| `tv.play()` | Start/resume playback |
| `tv.pause()` | Pause playback |
| `tv.stop()` | Stop playback |
| `tv.seek(track, time)` | Seek to position |
| `tv.get_inputs()` | Available input sources |
| `tv.set_input(uri)` | Switch input source |
| `tv.press_key(key)` | Press navigation key (async) |
| `tv.navigate(dir)` | Navigate UP/DOWN/LEFT/RIGHT/SELECT (async) |
| `tv.go_home()` | Press HOME button (async) |
| `tv.go_back()` | Press BACK button (async) |
| `tv.cast_url(url, type)` | Cast media URL (async) |
| `tv.cast_youtube(id)` | Cast YouTube video (async) |
| `tv.cast_spotify(uri)` | Cast Spotify URI (async) |
| `tv.launch_app(id)` | Launch Cast app (async) |
| `tv.stop_casting()` | Stop current Cast session (async) |

### Supported Keys

`UP`, `DOWN`, `LEFT`, `RIGHT`, `SELECT`, `BACK`, `HOME`, `RECENT`,
`PLAY_PAUSE`, `VOLUME_UP`, `VOLUME_DOWN`, `MUTE`, `INPUT`, `INFO`, `SETTINGS`

## Network Discovery

The skill uses SSDP (Simple Service Discovery Protocol) to find devices:

1. Sends multicast M-SEARCH to `239.255.255.250:1900`
2. Parses SSDP responses for Google TV/Chromecast signatures
3. Fetches device descriptor XML for detailed info
4. Identifies devices by:
   - `DiscoverFriendlies` UPnP service (Google TV specific)
   - Server header: `Linux/5.15.*` (Google TV firmware)
   - Model/friendly name containing "google", "chromecast", etc.

## Control Methods

The skill automatically selects the best control method:

1. **Google Cast API** (preferred) - WebSocket-based, supports key presses and media casting
2. **UPnP/DLNA** (fallback) - SOAP-based, supports volume and playback control
3. **None** - Device not reachable

## Files

| File | Purpose |
|---|---|
| `google_tv.py` | Main controller, unified interface |
| `scanner.py` | SSDP network discovery |
| `upnp_controller.py` | UPnP/DLNA control layer |
| `cast_controller.py` | Google Cast WebSocket API |

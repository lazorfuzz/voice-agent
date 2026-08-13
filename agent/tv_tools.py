"""Backend for Kronik's TV voice tools — reliable Google/Android TV (Cast) control
via pychromecast. Connection is cached to disk (connect-by-host, ~2s) so voice stays
snappy; falls back to mDNS discovery. Fuzzy content matching via YouTube search
(handles STT mishearings, e.g. "tempting island" -> Temptation Island)."""
import os, re, json, time, difflib, threading, subprocess, shutil
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, not ~/voice-agent
_CACHE = os.path.join(_ROOT, "tv_device.json")
_DEBUG = os.path.join(_ROOT, "tv_debug.log")
_REMOTE_CERT = os.path.join(_ROOT, "tv_remote_cert.pem")
_REMOTE_KEY = os.path.join(_ROOT, "tv_remote_key.pem")
_lock = threading.Lock()
_CAST = None  # cached live connection for this process


def _log(msg):
    try:
        with open(_DEBUG, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} [pid {os.getpid()}] {msg}\n")
    except Exception:
        pass

# Cast receiver app IDs for launching named apps
APP_IDS = {
    "youtube": "233637DE",
    "netflix": "CA5E8412",
    "spotify": "CC32E753",
    "disney plus": "C3DE6BC2",
    "hbo max": "b0b93c11-2e6a-4d1e-83e2-736c4a1f4a3e",
    "max": "b0b93c11-2e6a-4d1e-83e2-736c4a1f4a3e",
    "hulu": "3EC252A5",
    "prime video": "17608BC8",
    "plex": "9AC194DC",
}
_APP_ALIASES = {"disney": "disney plus", "disney+": "disney plus", "hbo": "hbo max",
                "amazon": "prime video", "amazon prime": "prime video", "prime": "prime video",
                "yt": "youtube", "you tube": "youtube"}


def _save_cache(cc):
    try:
        info = cc.cast_info
        json.dump({"host": info.host, "port": info.port, "uuid": str(info.uuid),
                   "model": info.model_name, "name": info.friendly_name}, open(_CACHE, "w"))
    except Exception:
        pass


def _discover_info():
    """One-time mDNS discovery to learn the TV's host/uuid; cache it to disk."""
    import pychromecast
    ccs, browser = pychromecast.get_chromecasts(timeout=12)
    try:
        if not ccs:
            raise RuntimeError("no TV found on the network")
        ci = ccs[0].cast_info
        info = {"host": ci.host, "port": ci.port, "uuid": str(ci.uuid),
                "model": ci.model_name, "name": ci.friendly_name,
                "cast_type": ci.cast_type, "manufacturer": ci.manufacturer}
        json.dump(info, open(_CACHE, "w"))
        return info
    finally:
        for c in ccs:
            try:
                c.disconnect(blocking=False)
            except Exception:
                pass
        try:
            pychromecast.discovery.stop_discovery(browser)
        except Exception:
            pass


def _connect_host(info):
    """Genuinely zeroconf-free connection by host (robust in a long-lived process)."""
    import uuid as _uuid
    from pychromecast import Chromecast
    from pychromecast.models import CastInfo, HostServiceInfo
    services = {HostServiceInfo(info["host"], info["port"])}
    ci = CastInfo(services, _uuid.UUID(info["uuid"]), info.get("model"), info.get("name"),
                  info["host"], info["port"], info.get("cast_type") or "cast",
                  info.get("manufacturer"))
    cc = Chromecast(ci, tries=2, timeout=10, retry_wait=2, zconf=None)
    cc.wait(timeout=12)
    return cc


def _connect():
    """Persistent host-based connection (no zeroconf dependency). Uses cached host
    for speed; re-discovers if the cache is stale (e.g. the TV's IP changed)."""
    info = None
    if os.path.exists(_CACHE):
        try:
            info = json.load(open(_CACHE))
        except Exception:
            info = None
    if info:
        try:
            return _connect_host(info)
        except Exception:
            try:
                os.remove(_CACHE)
            except Exception:
                pass
    return _connect_host(_discover_info())


def _get():
    """Cached live connection for this process, reconnecting if needed."""
    global _CAST
    with _lock:
        if _CAST is not None:
            try:
                _ = _CAST.status  # cheap liveness poke
                return _CAST
            except Exception:
                _CAST = None
        _CAST = _connect()
        return _CAST


# ---------------- controls ----------------
def control(action: str, value=None) -> str:
    _log(f"control(action={action!r}, value={value!r})")
    try:
        cc = _get()
    except Exception as e:
        return f"I couldn't reach the TV ({e}). Is it on?"
    mc = cc.media_controller
    a = (action or "").lower().strip().replace(" ", "_").replace("-", "_")
    a = {"turn_up": "volume_up", "turn_down": "volume_down", "louder": "volume_up",
         "quieter": "volume_down", "resume": "play"}.get(a, a)
    try:
        if a in ("volume_up", "up"):
            cc.volume_up(); return f"Turned it up — volume {round(cc.status.volume_level*100)}."
        if a in ("volume_down", "down"):
            cc.volume_down(); return f"Turned it down — volume {round(cc.status.volume_level*100)}."
        if a in ("set_volume", "volume"):
            lvl = max(0, min(100, int(value)))
            cc.set_volume(lvl / 100.0); return f"Volume set to {lvl}."
        if a == "mute":
            cc.set_volume_muted(True); return "Muted."
        if a == "unmute":
            cc.set_volume_muted(False); return "Unmuted."
        if a in ("pause",):
            mc.pause(); return "Paused."
        if a in ("play", "resume"):
            mc.play(); return "Playing."
        if a in ("stop",):
            mc.stop(); return "Stopped."
        if a in ("status", "what"):
            s = cc.status
            playing = mc.status.title or s.display_name or "nothing"
            return f"The TV is on {s.display_name or 'home'}, volume {round(s.volume_level*100)}, playing {playing}."
        return f"I don't know how to '{action}' the TV."
    except Exception as e:
        return f"That TV command failed ({type(e).__name__})."


# ---------------- Android TV control via ADB ----------------
# Why ADB: Google Cast can only LAUNCH an app (not deep-link a title), and the Android
# TV Remote protocol can't type into apps' custom search fields (Netflix ignores both
# `input text` and its own deep-links without Google's private intent extras). ADB
# (enabled once in the TV's Developer options -> USB/Network debugging, authorized once)
# is the reliable path: Google TV's GLOBAL_SEARCH intent jumps STRAIGHT to a title's
# detail page with the "Watch now" button focused; one D-pad center plays it on whatever
# service carries it (Netflix for Netflix titles, Prime, etc.). Google's search is
# fuzzy-tolerant, so mis-heard titles ("tempting island") still resolve correctly.

_ADB = shutil.which("adb") or "/opt/homebrew/bin/adb"


def _remote_host() -> str:
    try:
        return json.load(open(_CACHE))["host"]
    except Exception:
        return os.environ.get("TV_HOST", "")


def _adb_addr() -> str:
    return f"{_remote_host()}:5555"


def _adb(*args, timeout=25):
    return subprocess.run([_ADB, "-s", _adb_addr(), *args],
                          capture_output=True, text=True, timeout=timeout)


def adb_available() -> bool:
    """Ensure a connection (idempotent) and confirm the TV is authorized."""
    try:
        subprocess.run([_ADB, "connect", _adb_addr()],
                       capture_output=True, text=True, timeout=12)
        r = subprocess.run([_ADB, "devices"], capture_output=True, text=True, timeout=8)
        return f"{_adb_addr()}\tdevice" in r.stdout
    except Exception:
        return False


def _adb_focused() -> str:
    try:
        return _adb("shell", "dumpsys", "window", timeout=8).stdout
    except Exception:
        return ""


# Label fragments Google TV uses for each service (in tile content-desc or the
# "Recommended by <service>" row header), keyed by our normalized service name.
_SERVICE_LABELS = {
    "netflix": ["netflix"],
    "prime video": ["prime video", "prime"],
    "disney plus": ["disney+", "disney plus", "disney"],
    "hbo max": ["hbo max", "max"],
    "max": ["hbo max", "max"],
    "hulu": ["hulu"],
    "apple tv": ["apple tv"],
    "peacock": ["peacock"],
    "paramount plus": ["paramount+", "paramount"],
}

_NODE_RE = re.compile(r"<node\b([^>]*?)/?>")


def _sanitize(query: str) -> str:
    """Keep only characters that are safe to pass through `adb shell input text`."""
    q = re.sub(r"[^A-Za-z0-9 ]+", " ", query)
    return re.sub(r"\s+", " ", q).strip()


def _adb_dump() -> str:
    """uiautomator view hierarchy XML of whatever is currently on screen."""
    try:
        out = _adb("shell", "uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; "
                            "cat /sdcard/ui.xml", timeout=15).stdout
        i = out.find("<?xml")
        return out[i:] if i >= 0 else out
    except Exception:
        return ""


def _attr(node: str, name: str) -> str:
    m = re.search(rf'{name}="([^"]*)"', node)
    return m.group(1) if m else ""


def _parse_nodes(xml: str):
    nodes = []
    for m in _NODE_RE.finditer(xml):
        s = m.group(1)
        b = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', s)
        if not b:
            continue
        nodes.append({
            "text": _attr(s, "text"),
            "desc": _attr(s, "content-desc"),
            "cls": _attr(s, "class"),
            "clickable": _attr(s, "clickable") == "true",
            "focused": _attr(s, "focused") == "true",
            "bounds": (int(b.group(1)), int(b.group(2)), int(b.group(3)), int(b.group(4))),
        })
    return nodes


def _center(b):
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _field_text() -> str:
    for n in _parse_nodes(_adb_dump()):
        if "EditText" in n["cls"]:
            return n["text"]
    return ""


def _type_query(q: str) -> None:
    """Type `q` into the Google TV search box. The FIRST `input text` only ACTIVATES the
    field (its content is silently dropped), so type once to warm it, clear, then type for
    real — the warmed field reliably accepts the second type. No read-back/verify needed,
    which keeps it snappy. Spaces are backslash-escaped (quoting and %s get eaten)."""
    esc = q.replace(" ", "\\ ")
    _adb("shell", f"input text {esc}")             # activate (dropped)
    time.sleep(0.3)
    _adb("shell", "input keyevent" + " 67" * 30)   # clear
    _adb("shell", f"input text {esc}")             # real type -> lands
    time.sleep(0.3)


def _title_match(q: str, desc: str) -> bool:
    title = desc.split(",")[0].strip().lower()
    ql = q.strip().lower()
    if not title:
        return False
    if ql in title or title in ql:
        return True
    return difflib.SequenceMatcher(None, ql, title).ratio() >= 0.7


def _pick_service_tile(q: str, labels):
    """On the search-results screen, find the tile for `q` that belongs to the wanted
    service — either its own content-desc names the service, or the nearest row header
    above it says "Recommended by <service>". Returns tap (x, y) or None."""
    nodes = _parse_nodes(_adb_dump())
    headers = [(n["text"], n["bounds"][1]) for n in nodes
               if n["text"] and not n["clickable"] and "TextView" in n["cls"]]
    tiles = [n for n in nodes if n["clickable"] and n["desc"] and _title_match(q, n["desc"])]
    for t in tiles:                                     # tile's own label names the service
        d = t["desc"].lower()
        if any(l in d for l in labels):
            return _center(t["bounds"])
    for t in tiles:                                     # else the row header above it
        ty = t["bounds"][1]
        above = [h for h in headers if h[1] < ty]
        if above and any(l in max(above, key=lambda h: h[1])[0].lower() for l in labels):
            return _center(t["bounds"])
    return None


def _is_play_button(n) -> bool:
    d = n["desc"].lower()
    return n["clickable"] and ("watch now" in d or "resume" in d or d.endswith(", play"))


def _media_playing() -> bool:
    """True if a media session reports active playback (PlaybackState 3=playing,
    4=ff, 5=rew, 6=buffering)."""
    try:
        out = _adb("shell", "dumpsys media_session", timeout=8).stdout
        return bool(re.search(r"state=PlaybackState \{state=[3456]\b", out))
    except Exception:
        return False


def _nudge_play(settle: float = 5.0, tries: int = 4, gap: float = 2.5) -> None:
    """After a streaming app opens on a title, press SELECT to start it if it hasn't
    autoplayed (Netflix often lands on the title's Play screen and just waits). Gated on
    the media-session state so we NEVER press while something is already playing — a
    center press on a playing video would pause it. Netflix is a DRM black box, so the
    media session is the only readable signal."""
    time.sleep(settle)
    for _ in range(tries):
        if _media_playing():
            return
        _adb("shell", "input keyevent KEYCODE_DPAD_CENTER")
        time.sleep(gap)


def _global_search_play(q: str) -> bool:
    """GLOBAL_SEARCH jumps to the top match's detail page. Wait until the "Watch now"
    button is rendered AND focused, then D-pad SELECT it. Never input-tap TV detail
    buttons — a touch steals D-pad focus and the button goes dead; and a blind center
    right after the activity opens races the render and lands on the page without playing.

    Returns True when a playable entity was found and selected (or media is playing);
    False when the search surfaced nothing playable (e.g. YouTube-channel content that
    isn't in the Google TV catalog) so the caller can fall back to a YouTube cast."""
    _adb("shell", "input keyevent KEYCODE_HOME")
    time.sleep(0.6)
    _adb("shell", f'am start -a android.search.action.GLOBAL_SEARCH --es query "{q}"')
    entity = False
    for _ in range(16):
        time.sleep(0.4)
        if "EntityActivity" in _adb_focused():
            entity = True
            break
    if not entity:
        return False                          # no detail page = nothing to play here
    pressed = False
    for _ in range(10):                      # wait for Watch now to render + take focus
        time.sleep(0.5)
        nodes = _parse_nodes(_adb_dump())
        if any(_is_play_button(n) and n["focused"] for n in nodes):
            _adb("shell", "input keyevent KEYCODE_DPAD_CENTER")   # select -> launches app
            pressed = True
            break
    else:
        _adb("shell", "input keyevent KEYCODE_DPAD_CENTER")   # fallback
    _nudge_play()   # make sure the streaming app actually starts playing
    return pressed or _media_playing()


def adb_play_title(query: str, service: str = None) -> str:
    """Play `query` on the TV. If `service` is given (e.g. "netflix"), open Google TV
    search and pick that service's result so the requested app actually plays; otherwise
    jump to the top match and play it on whatever app carries it."""
    _log(f"adb_play_title(query={query!r}, service={service!r})")
    if not adb_available():
        return ("I can't reach the TV over ADB. Make sure it's on and USB/network "
                "debugging is enabled.")
    q = _sanitize(query)
    labels = _SERVICE_LABELS.get(service) if service else None
    try:
        # Honor a specific service via the results list + tile selection.
        if labels:
            _adb("shell", "input keyevent KEYCODE_HOME")
            time.sleep(0.7)
            _adb("shell", "input tap 276 108")            # Search icon (stable position)
            time.sleep(1.0)
            _type_query(q)
            _adb("shell", "input keyevent 66")           # ENTER -> results list
            time.sleep(1.2)
            xy = _pick_service_tile(q, labels)
            if xy:
                _adb("shell", f"input tap {xy[0]} {xy[1]}")   # launches the app
                _nudge_play()                                 # press play if it doesn't autoplay
                _log(f"adb_play_title OK service={service} xy={xy}")
                return f"Playing '{query}' on {service.title()}."
            _log("adb_play_title: no service tile; falling back to top result")

        # No service (or the service didn't have it) -> top match on its default app.
        if _global_search_play(q):
            _log(f"adb_play_title OK generic q={q!r}")
            return (f"Playing '{query}' on {service.title()}." if service
                    else f"Playing '{query}' on the TV.")
        # Catalog search came up dry (typical for YouTuber/channel videos, e.g. a "fern
        # documentary" — live miss 2026-07-25) -> cast the top YouTube result instead.
        _log("global search: nothing playable — falling back to YouTube cast")
        res = _cast_youtube(query)
        if res.startswith("Playing"):
            return res
        return (f"I searched the TV for '{query}' but couldn't find anything playable, "
                f"and the YouTube fallback failed too ({res})")
    except Exception as e:
        _log(f"adb_play_title ERROR {type(e).__name__}: {e}")
        return f"Couldn't start '{query}' on the TV ({type(e).__name__})."


# ---------------- fuzzy content ----------------
def _fuzzy_app(name: str):
    if not name:
        return None
    n = name.lower().strip()
    n = _APP_ALIASES.get(n, n)
    keys = list(APP_IDS.keys())
    if n in APP_IDS:
        return n
    m = difflib.get_close_matches(n, keys + list(_APP_ALIASES.keys()), n=1, cutoff=0.6)
    if m:
        return _APP_ALIASES.get(m[0], m[0])
    return None


def _youtube_search(query: str):
    """Return (video_id, title) for the top YouTube result. YouTube's search is
    fuzzy-tolerant, so a misheard query still resolves to the right video."""
    import urllib.request, urllib.parse
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en"})
    html = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "replace")
    m = re.search(r'"videoId":"([\w-]{11})"', html)
    if not m:
        return None, query
    vid = m.group(1)
    tm = re.search(r'"videoId":"' + re.escape(vid) + r'".*?"text":"([^"]{2,100})"', html, re.S)
    return vid, (tm.group(1) if tm else query)


_YT = {"cc": None, "ctl": None}


def _yt_play(cc, vid):
    """ONE YouTubeController per cast connection. Registering a fresh controller on the
    cached connection for every play breaks the mdx screen_id handshake after the first
    cast — every later cast dies with Lounge API 400 "screen_ids parameter error"
    (seen live 2026-07-25: first video played, all retries silently failed)."""
    from pychromecast.controllers.youtube import YouTubeController
    if _YT["ctl"] is None or _YT["cc"] is not cc:
        ctl = YouTubeController()
        cc.register_handler(ctl)
        _YT.update(cc=cc, ctl=ctl)
    _YT["ctl"].play_video(vid)


def _cast_youtube(query: str) -> str:
    """Search YouTube + cast the top result. Used for explicit service="youtube"
    requests AND as the fallback when Google TV's catalog search has nothing playable
    (YouTuber/channel content lives on YouTube, not in the catalog)."""
    global _CAST
    try:
        cc = _get()
    except Exception as e:
        return f"I couldn't reach the TV ({e}). Is it on?"
    try:
        vid, title = _youtube_search(query)
        if not vid:
            return f"Couldn't find '{query}' on YouTube."
        # If the YouTube app is ALREADY up (e.g. something is playing), the mdx
        # screen_id handshake never answers and the cast 400s — quit it first and
        # let play_video relaunch it clean. (Live failure mode 2026-07-25: first
        # cast of the evening worked, every one after it failed.)
        try:
            if (cc.app_id or "") == "233637DE":      # YouTube receiver app id
                _log("youtube app already running — quitting it before cast")
                cc.quit_app()
                time.sleep(2.5)
                _YT["cc"] = _YT["ctl"] = None
        except Exception as e:
            _log(f"quit_app failed (continuing): {e}")
        for attempt in (1, 2):
            try:
                _yt_play(cc, vid)
                _log(f"youtube cast OK vid={vid} title={title!r}")
                return f"Playing '{title}' on YouTube."
            except Exception as e:
                _log(f"youtube attempt {attempt} {type(e).__name__}: {e}")
                if attempt == 2:
                    raise
                # stale cast session: drop connection + controller, reconnect once
                with _lock:
                    _CAST = None
                _YT["cc"] = _YT["ctl"] = None
                cc = _get()
    except Exception as e:
        _log(f"youtube ERROR {type(e).__name__}: {e}")
        return f"Couldn't start YouTube ({type(e).__name__})."


def play(query: str, service: str = None) -> str:
    _log(f"play(query={query!r}, service={service!r})")
    app = _fuzzy_app(service) if service else None

    # Explicit YouTube -> fuzzy search + Cast. Best for music, clips, and specific
    # videos (Cast starts them instantly and YouTube search tolerates mishearings).
    if app == "youtube":
        return _cast_youtube(query)

    # Any show/movie — a named streaming service (Netflix, Prime, Disney+, Hulu, Max…)
    # OR no service given — goes through Google TV. With a named service we pick that
    # service's result so the right app plays; with none we play the top match.
    svc = app or (service.strip().lower() if service else None)
    return adb_play_title(query, svc)

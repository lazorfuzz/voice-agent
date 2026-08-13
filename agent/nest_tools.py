"""Backend for Kronik's Nest thermostat tools via Google's Smart Device Management (SDM)
cloud API. Nest exposes no local API, so this is the official supported path. Credentials
(OAuth client + long-lived refresh token + device id) live in .env.local; the short-lived
access token is refreshed automatically and cached."""
import os
import json
import time
import threading
import urllib.request
import urllib.parse

_PROJECT = os.environ.get("NEST_PROJECT_ID", "")
_CLIENT_ID = os.environ.get("NEST_CLIENT_ID", "")
_CLIENT_SECRET = os.environ.get("NEST_CLIENT_SECRET", "")
_REFRESH = os.environ.get("NEST_REFRESH_TOKEN", "")
_DEVICE = os.environ.get("NEST_DEVICE_ID", "")
_SDM = "https://smartdevicemanagement.googleapis.com/v1"

_lock = threading.Lock()
_token = {"value": None, "exp": 0.0}


def _c2f(c):
    return round(c * 9 / 5 + 32) if isinstance(c, (int, float)) else c


def _f2c(f):
    return round((f - 32) * 5 / 9, 2)


def _access_token():
    with _lock:
        if _token["value"] and time.time() < _token["exp"] - 60:
            return _token["value"]
        data = urllib.parse.urlencode({
            "client_id": _CLIENT_ID, "client_secret": _CLIENT_SECRET,
            "refresh_token": _REFRESH, "grant_type": "refresh_token",
        }).encode()
        r = json.load(urllib.request.urlopen(
            urllib.request.Request("https://oauth2.googleapis.com/token", data=data), timeout=15))
        _token["value"] = r["access_token"]
        _token["exp"] = time.time() + r.get("expires_in", 3600)
        return _token["value"]


def _api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{_SDM}/enterprises/{_PROJECT}/{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {_access_token()}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def _cmd(command, params):
    return _api("POST", f"devices/{_DEVICE}:executeCommand", {"command": command, "params": params})


def _traits():
    return _api("GET", f"devices/{_DEVICE}").get("traits", {})


def _t(tr, name, key, d=None):
    return tr.get("sdm.devices.traits." + name, {}).get(key, d)


_MODE_ALIASES = {
    "heat": "HEAT", "heating": "HEAT", "warm": "HEAT", "warmer": "HEAT",
    "cool": "COOL", "cooling": "COOL", "ac": "COOL", "a/c": "COOL", "air": "COOL",
    "air conditioning": "COOL", "cold": "COOL", "colder": "COOL",
    "auto": "HEATCOOL", "heatcool": "HEATCOOL", "heat/cool": "HEATCOOL",
    "heat-cool": "HEATCOOL", "range": "HEATCOOL", "off": "OFF", "stop": "OFF",
}
_CMD = "sdm.devices.commands.ThermostatTemperatureSetpoint."


def status() -> str:
    try:
        tr = _traits()
    except Exception as e:
        return f"I couldn't reach the thermostat ({type(e).__name__}). Is it online?"
    mode = _t(tr, "ThermostatMode", "mode")
    hvac = _t(tr, "ThermostatHvac", "status")
    amb = _t(tr, "Temperature", "ambientTemperatureCelsius")
    hum = _t(tr, "Humidity", "ambientHumidityPercent")
    sp = tr.get("sdm.devices.traits.ThermostatTemperatureSetpoint", {})
    heat, cool = sp.get("heatCelsius"), sp.get("coolCelsius")
    lead = f"It's {_c2f(amb)} degrees" + (f" with {hum}% humidity" if hum is not None else "")
    if mode == "HEAT" and heat is not None:
        target = f"set to heat to {_c2f(heat)}"
    elif mode == "COOL" and cool is not None:
        target = f"set to cool to {_c2f(cool)}"
    elif mode == "HEATCOOL" and heat is not None and cool is not None:
        target = f"on auto, holding between {_c2f(heat)} and {_c2f(cool)}"
    elif mode == "OFF":
        target = "and the thermostat is off"
    else:
        target = f"in {mode} mode"
    action = {"HEATING": "currently heating", "COOLING": "currently cooling"}.get(hvac)
    return ", ".join([lead, target] + ([action] if action else [])) + "."


def set_mode(mode: str) -> str:
    m = _MODE_ALIASES.get((mode or "").lower().strip())
    if not m:
        return f"I don't know the mode '{mode}'. Try heat, cool, auto, or off."
    try:
        _cmd("sdm.devices.commands.ThermostatMode.SetMode", {"mode": m})
        return {"HEAT": "Switched it to heat.", "COOL": "Switched it to cool.",
                "HEATCOOL": "Set it to auto — it'll heat and cool as needed.",
                "OFF": "Turned the thermostat off."}[m]
    except Exception as e:
        return f"Couldn't change the mode ({type(e).__name__})."


def set_temperature(temp_f, target: str = None) -> str:
    try:
        tr = _traits()
    except Exception as e:
        return f"I couldn't reach the thermostat ({type(e).__name__})."
    try:
        temp_f = float(temp_f)
    except Exception:
        return "That temperature didn't make sense."
    mode = _t(tr, "ThermostatMode", "mode")
    hvac = _t(tr, "ThermostatHvac", "status")
    sp = tr.get("sdm.devices.traits.ThermostatTemperatureSetpoint", {})
    heat, cool = sp.get("heatCelsius"), sp.get("coolCelsius")
    c = _f2c(temp_f)
    k = (target or "").lower().strip()
    try:
        if mode == "HEAT":
            _cmd(_CMD + "SetHeat", {"heatCelsius": c})
            return f"Set the heat to {round(temp_f)}."
        if mode == "COOL":
            _cmd(_CMD + "SetCool", {"coolCelsius": c})
            return f"Set the cooling to {round(temp_f)}."
        if mode == "HEATCOOL":
            # auto mode has two setpoints; use the caller's hint, else the side that's active
            side = k if k in ("heat", "cool") else {"HEATING": "heat", "COOLING": "cool"}.get(hvac)
            if side == "heat":
                _cmd(_CMD + "SetRange", {"heatCelsius": c, "coolCelsius": cool})
                return f"Set the heat target to {round(temp_f)}."
            if side == "cool":
                _cmd(_CMD + "SetRange", {"heatCelsius": heat, "coolCelsius": c})
                return f"Set the cool target to {round(temp_f)}."
            return ("It's on auto with a heat and a cool target — should I set that as the heat "
                    "or the cool temperature?")
        return "The thermostat's off — switch it to heat or cool first, then I can set a temperature."
    except Exception as e:
        return f"Couldn't set the temperature ({type(e).__name__}) — it may be out of the allowed range."

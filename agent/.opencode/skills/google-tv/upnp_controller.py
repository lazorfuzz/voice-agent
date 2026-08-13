"""
UPnP/DLNA Controller for Google TV.

Controls Google Smart TVs via UPnP/DLNA protocols (AVTransport,
RenderingControl, ConnectionManager services).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from scanner import DeviceInfo


@dataclass
class TransportInfo:
    """Current transport state information."""

    current_state: str = "STOPPED"
    transport_status: str = "OK"
    play_mode: str = "NORMAL"
    cur_track: int = 0
    cur_track_duration: str = ""
    cur_track_metadata: str = ""
    cur_track_meta_data: str = ""
    next_track: int = 0
    next_track_duration: str = ""
    next_track_metadata: str = ""
    next_track_meta_data: str = ""
    track_change_time: str = ""


@dataclass
class VolumeInfo:
    """Current volume and mute state."""

    volume: int = 50
    mute: bool = False
    volume_db: float = 0.0


class UPnPController:
    """Controls a Google TV via UPnP/DLNA protocols."""

    def __init__(self, device: DeviceInfo):
        self.device = device
        self.av_transport_url = device.get_control_url(
            "urn:schemas-upnp-org:service:AVTransport:1"
        )
        self.rendering_control_url = device.get_control_url(
            "urn:schemas-upnp-org:service:RenderingControl:1"
        )
        self.connection_manager_url = device.get_control_url(
            "urn:schemas-upnp-org:service:ConnectionManager:1"
        )

    def _service_type_for(self, control_url: str) -> str:
        """Reverse-lookup the UPnP service type (needed for SOAPAction/namespace)."""
        for stype, url in self.device.control_urls.items():
            if url == control_url:
                return stype
        return "urn:schemas-upnp-org:service:AVTransport:1"

    def _send_soap_request(
        self, control_url: str, action: str, body: dict
    ) -> Optional[ET.Element]:
        """Send a correctly-formed UPnP SOAP request and return the parsed response."""
        import urllib.request

        service_type = self._service_type_for(control_url)
        # AVTransport/RenderingControl actions require InstanceID first.
        params = {"InstanceID": "0"}
        params.update(body)
        args = "".join(f"<{k}>{v}</{k}>" for k, v in params.items())
        soap_envelope = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
            's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
            "<s:Body>"
            f'<u:{action} xmlns:u="{service_type}">{args}</u:{action}>'
            "</s:Body></s:Envelope>"
        )
        try:
            req = urllib.request.Request(control_url, data=soap_envelope.encode("utf-8"))
            req.add_header("Content-Type", 'text/xml; charset="utf-8"')
            req.add_header("SOAPACTION", f'"{service_type}#{action}"')
            req.add_header("Connection", "close")
            resp = urllib.request.urlopen(req, timeout=10)
            return ET.fromstring(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return None

    def get_transport_state(self) -> TransportInfo:
        """Get the current transport state of the TV."""
        info = TransportInfo()
        if not self.av_transport_url:
            return info

        result = self._send_soap_request(
            self.av_transport_url, "GetTransportInfo", {}
        )
        if result is None:
            return info
        el = result.find(".//{*}CurrentTransportState")
        if el is not None and el.text:
            info.current_state = el.text.strip()
        el = result.find(".//{*}CurrentTransportStatus")
        if el is not None and el.text:
            info.transport_status = el.text.strip()
        return info

    def get_position_info(self) -> dict:
        """Get current playback position information."""
        if not self.av_transport_url:
            return {}

        result = self._send_soap_request(
            self.av_transport_url, "GetPositionInfo", {}
        )
        if result is None:
            return {}

        body = result.find(".//{*}Body")
        if body is None:
            body = result.find(".//Body")

        if body is not None:
            elem = body.find(
                ".//{*}GetPositionInfoResult"
            )
            if elem is None:
                elem = body.find(".//GetPositionInfoResult")

            if elem is not None:
                return {
                    "track": int(
                        _get_xml_text(elem, "Track", "0") or "0"
                    ),
                    "track_duration": _get_xml_text(
                        elem, "TrackDuration", ""
                    ),
                    "rel_time": _get_xml_text(elem, "RelTime", ""),
                    "abs_time": _get_xml_text(elem, "AbsTime", ""),
                    "rel_count": _get_xml_text(
                        elem, "RelCount", ""
                    ),
                    "abs_count": _get_xml_text(
                        elem, "AbsCount", ""
                    ),
                }

        return {}

    def get_device_volume(self) -> VolumeInfo:
        """Get the current volume and mute state."""
        info = VolumeInfo()
        if not self.rendering_control_url:
            return info

        result = self._send_soap_request(
            self.rendering_control_url, "GetVolume", {"Channel": "Master"}
        )
        if result is not None:
            elem = result.find(".//{*}CurrentVolume")
            if elem is not None and elem.text:
                try:
                    info.volume = int(elem.text.strip())
                except ValueError:
                    pass

        result = self._send_soap_request(
            self.rendering_control_url, "GetMute", {"Channel": "Master"}
        )
        if result is not None:
            elem = result.find(".//{*}CurrentMute")
            if elem is not None and elem.text:
                info.mute = elem.text.strip() in ("1", "true", "True")

        return info

    def set_volume(self, volume: int) -> bool:
        """Set the TV volume (0-100)."""
        if not self.rendering_control_url:
            return False

        result = self._send_soap_request(
            self.rendering_control_url,
            "SetVolume",
            {"Channel": "Master", "DesiredVolume": str(volume)},
        )
        return result is not None

    def set_mute(self, mute: bool) -> bool:
        """Mute or unmute the TV."""
        if not self.rendering_control_url:
            return False

        result = self._send_soap_request(
            self.rendering_control_url,
            "SetMute",
            {
                "Channel": "Master",
                "DesiredMute": "1" if mute else "0",
            },
        )
        return result is not None

    def volume_up(self) -> bool:
        """Increase volume by 1."""
        return self.set_volume(
            min(self.get_device_volume().volume + 1, 100)
        )

    def volume_down(self) -> bool:
        """Decrease volume by 1."""
        return self.set_volume(
            max(self.get_device_volume().volume - 1, 0)
        )

    def play(self) -> bool:
        """Start/resume playback."""
        if not self.av_transport_url:
            return False

        result = self._send_soap_request(
            self.av_transport_url, "Play", {"Speed": "1"}
        )
        return result is not None

    def pause(self) -> bool:
        """Pause playback."""
        if not self.av_transport_url:
            return False

        result = self._send_soap_request(
            self.av_transport_url, "Pause", {}
        )
        return result is not None

    def stop(self) -> bool:
        """Stop playback."""
        if not self.av_transport_url:
            return False

        result = self._send_soap_request(
            self.av_transport_url, "Stop", {}
        )
        return result is not None

    def seek(self, track_number: int = 0, track_time: str = "") -> bool:
        """Seek to a specific position."""
        if not self.av_transport_url:
            return False

        result = self._send_soap_request(
            self.av_transport_url,
            "Seek",
            {
                "Unit": "track-number" if track_number else "relTime",
                "Target": str(track_number) if track_number else track_time,
            },
        )
        return result is not None

    def set_transport_url(self, uri: str, metadata: str = "") -> bool:
        """Set the transport URL (for casting media)."""
        if not self.av_transport_url:
            return False

        result = self._send_soap_request(
            self.av_transport_url,
            "SetAVTransportURI",
            {"CurrentURI": uri, "CurrentURIMetaData": metadata},
        )
        return result is not None

    def get_current_uri(self) -> dict:
        """Get the currently playing URI information."""
        if not self.av_transport_url:
            return {}

        result = self._send_soap_request(
            self.av_transport_url, "GetCurrentURI", {}
        )
        if result is None:
            return {}

        body = result.find(".//{*}Body")
        if body is None:
            body = result.find(".//Body")

        if body is not None:
            elem = body.find(".//{*}GetCurrentURIResult")
            if elem is None:
                elem = body.find(".//GetCurrentURIResult")

            if elem is not None:
                return {
                    "current_uri": _get_xml_text(
                        elem, "CurrentURI", ""
                    ),
                    "current_urimeta": _get_xml_text(
                        elem, "CurrentURIMetaData", ""
                    ),
                }

        return {}

    def get_available_inputs(self) -> list[dict]:
        """Get available input sources."""
        inputs = []
        if not self.av_transport_url:
            return inputs

        result = self._send_soap_request(
            self.av_transport_url, "GetMediaInfo", {}
        )
        if result is None:
            return inputs

        body = result.find(".//{*}Body")
        if body is None:
            body = result.find(".//Body")

        if body is not None:
            elem = body.find(".//{*}GetMediaInfoResult")
            if elem is None:
                elem = body.find(".//GetMediaInfoResult")

            if elem is not None:
                inputs.append(
                    {
                        "name": "current",
                        "uri": _get_xml_text(
                            elem, "CurrentURI", ""
                        ),
                        "track": int(
                            _get_xml_text(elem, "NrTracks", "0") or "0"
                        ),
                    }
                )

        return inputs

    def get_device_info(self) -> dict:
        """Get device information from the device descriptor."""
        info = {
            "friendly_name": self.device.friendly_name,
            "model_name": self.device.model_name,
            "manufacturer": self.device.manufacturer,
            "uuid": self.device.uuid,
            "services": self.device.services,
        }
        return info


def _build_xml_body(action: str, parameters: dict) -> str:
    """Build the XML body for a SOAP request."""
    params = "".join(
        f"<{k}>{v}</{k}" for k, v in parameters.items()
    )
    return f"<u:{action} xmlns:u=\"urn:schemas-upnp-org:service:{action}:1\">{params}</u:{action}>"


def _get_xml_text(
    element: ET.Element, tag: str, default: str = ""
) -> str:
    """Get text content of a child element."""
    child = element.find(f"{{{element.tag.split('}')[0] if '}' in element.tag else ''}}}{tag}")
    if child is None:
        child = element.find(tag)
    if child is None:
        return default
    return child.text.strip() if child.text else default


def create_upnp_controller(device: DeviceInfo) -> UPnPController:
    """Create a UPnPController for the given device."""
    return UPnPController(device)

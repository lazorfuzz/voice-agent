"""
SSDP Network Scanner for Google TV / Chromecast Discovery.

Discovers Google Smart TVs and Chromecast devices on the local network
using SSDP (Simple Service Discovery Protocol) multicast.
"""

import socket
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeviceInfo:
    """Represents a discovered Google TV / Chromecast device."""

    friendly_name: str = ""
    model_name: str = ""
    model_number: str = ""
    manufacturer: str = ""
    serial_number: str = ""
    uuid: str = ""
    device_type: str = ""
    location_url: str = ""
    control_urls: dict = field(default_factory=dict)
    event_sub_urls: dict = field(default_factory=dict)
    scpd_urls: dict = field(default_factory=dict)
    services: list = field(default_factory=list)
    response_headers: dict = field(default_factory=dict)
    source_ip: str = ""
    source_port: int = 0
    raw_xml: str = ""

    @property
    def is_google_tv(self) -> bool:
        """Check if the device is a Google TV."""
        google_indicators = [
            "google",
            "chromecast",
            "androidtv",
            "google tv",
            "googletv",
            "crkey",
            "nexus player",
        ]
        name_lower = (
            self.friendly_name + self.model_name + self.manufacturer
        ).lower()
        if any(indicator in name_lower for indicator in google_indicators):
            return True
        if "DiscoverFriendlies" in self.services:
            return True
        if "Linux/5.15" in self.response_headers.get("Server", ""):
            return True
        return False

    @property
    def is_chromecast(self) -> bool:
        """Check if the device is a Chromecast."""
        chromecast_indicators = [
            "chromecast",
            "crkey",
            "nexus player",
            "google cast",
        ]
        name_lower = (
            self.friendly_name + self.model_name + self.manufacturer
        ).lower()
        return any(indicator in name_lower for indicator in chromecast_indicators)

    @property
    def is_media_renderer(self) -> bool:
        """Check if the device is a UPnP Media Renderer."""
        return "MediaRenderer" in self.device_type

    @property
    def has_cast_api(self) -> bool:
        """Check if the device exposes a Cast API endpoint."""
        cast_ports = [8008, 8002, 8009, 8010]
        return any(
            self.location_url.startswith(f"http://{self.source_ip}:{p}")
            for p in cast_ports
        )

    def get_control_url(self, service_type: str) -> str:
        """Get the control URL for a given service type."""
        return self.control_urls.get(service_type, "")

    def get_base_url(self) -> str:
        """Get the base URL for the device."""
        return self.location_url.rsplit("/", 1)[0]


def _local_ip() -> str:
    """Best-effort primary LAN IPv4 (the interface toward the default gateway).
    On a multi-NIC host (e.g. with container vmnet/link-local interfaces) this
    ensures the multicast M-SEARCH egresses the real LAN, not a stray interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ""


def send_ssdp_search(mx_seconds: float = 2.0, iface_ip: str = "") -> list[DeviceInfo]:
    """
    Send SSDP M-SEARCH request and return discovered devices.

    Args:
        mx_seconds: Maximum wait time in seconds for responses.
        iface_ip: Local interface IP to send from (defaults to the primary LAN IP).

    Returns:
        List of discovered DeviceInfo objects.
    """
    ssdp_msearch = (
        f"M-SEARCH * HTTP/1.1\r\n"
        f"HOST: 239.255.255.250:1900\r\n"
        f'MAN: "ssdp:discover"\r\n'
        f"MX: {int(mx_seconds)}\r\n"
        f"ST: ssdp:all\r\n"
        f"USER-AGENT: OpenCode-GoogleTV-Scanner/1.0\r\n"
        f"\r\n"
    ).encode()

    local_ip = iface_ip or _local_ip()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if local_ip:
        try:
            sock.bind((local_ip, 0))  # egress the LAN interface
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(local_ip))
        except OSError:
            pass
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(mx_seconds + 1)

    devices: list[DeviceInfo] = []
    seen_uuids: set[str] = set()

    try:
        sock.sendto(ssdp_msearch, ("239.255.255.250", 1900))

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                response = data.decode("utf-8", errors="replace")
                parsed = _parse_ssdp_response(response, addr)
                if parsed and parsed.uuid not in seen_uuids:
                    seen_uuids.add(parsed.uuid)
                    devices.append(parsed)
            except socket.timeout:
                break
    finally:
        sock.close()

    return devices


def _parse_ssdp_response(
    response: str, source_addr: tuple[str, int]
) -> Optional[DeviceInfo]:
    """Parse an SSDP response into a DeviceInfo object."""
    lines = response.strip().split("\r\n")
    if not lines or "HTTP" not in lines[0]:
        return None

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()

    uuid = headers.get("USN", "")
    location = headers.get("Location", "")
    st = headers.get("ST", "")

    device = DeviceInfo(
        uuid=uuid.split("::")[0] if "::" in uuid else uuid,
        location_url=location,
        device_type=st,
        source_ip=source_addr[0],
        source_port=source_addr[1],
        response_headers=headers,
    )

    if location:
        device = _fetch_device_descriptor(device)

    return device


def _fetch_device_descriptor(device: DeviceInfo) -> DeviceInfo:
    """Fetch and parse the device descriptor XML from the Location URL."""
    import urllib.request
    from urllib.parse import urljoin

    try:
        # Fetch the actual descriptor XML at the Location URL (not its parent dir).
        req = urllib.request.Request(device.location_url)
        req.add_header("Connection", "close")
        resp = urllib.request.urlopen(req, timeout=5)
        xml_content = resp.read().decode("utf-8", errors="replace")
        device.raw_xml = xml_content

        root = ET.fromstring(xml_content)
        ns = {
            "upnp": "urn:schemas-upnp-org:device-1-0",
            "dlna": "urn:schemas-dlna-org:device-1-0",
        }

        device_elem = root.find("upnp:device", ns)
        if device_elem is None:
            device_elem = root.find(".//device")

        if device_elem is not None:
            device.friendly_name = _get_text(device_elem, "friendlyName", ns)
            device.model_name = _get_text(device_elem, "modelName", ns)
            device.model_number = _get_text(device_elem, "modelNumber", ns)
            device.manufacturer = _get_text(device_elem, "manufacturer", ns)
            device.serial_number = _get_text(
                device_elem, "serialNumber", ns
            )
            device.uuid = _get_text(device_elem, "UDN", ns).replace(
                "uuid:", ""
            )

            service_list = device_elem.find(
                "upnp:serviceList", ns
            )
            if service_list is None:
                service_list = device_elem.find(".//serviceList")

            if service_list is not None:
                for service in service_list.findall(
                    "upnp:service", ns
                ):
                    if service is None:
                        service = service_list.find(".//service")
                    if service is None:
                        continue
                    service_type = _get_text(service, "serviceType", ns)
                    service_id = _get_text(service, "serviceId", ns)
                    control_url = _get_text(service, "controlURL", ns)
                    event_sub_url = _get_text(
                        service, "eventSubURL", ns
                    )
                    scpd_url = _get_text(service, "SCPDURL", ns)

                    # Resolve service URLs against the descriptor Location (UPnP spec)
                    if service_type:
                        device.services.append(service_type)
                    if control_url:
                        device.control_urls[service_type] = urljoin(
                            device.location_url, control_url
                        )
                    if event_sub_url:
                        device.event_sub_urls[service_type] = urljoin(
                            device.location_url, event_sub_url
                        )
                    if scpd_url:
                        device.scpd_urls[service_type] = urljoin(
                            device.location_url, scpd_url
                        )

    except Exception:
        pass

    return device


def _get_text(
    element: ET.Element, tag: str, ns: dict[str, str]
) -> str:
    """Get text content of a child element, handling namespaces."""
    child = element.find(f"{{{ns.get('upnp', '')}}}{tag}")
    if child is None:
        child = element.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def discover_google_tvs(
    mx_seconds: float = 2.0,
) -> list[DeviceInfo]:
    """
    Discover all Google TVs and Chromecast devices on the network.

    Args:
        mx_seconds: Maximum wait time for SSDP responses.

    Returns:
        List of discovered Google TV / Chromecast DeviceInfo objects.
    """
    all_devices = send_ssdp_search(mx_seconds)
    google_devices = [
        d for d in all_devices if d.is_google_tv or d.is_chromecast
    ]
    return google_devices


def discover_all_devices(
    mx_seconds: float = 2.0,
) -> list[DeviceInfo]:
    """
    Discover all UPnP devices on the network.

    Args:
        mx_seconds: Maximum wait time for SSDP responses.

    Returns:
        List of all discovered DeviceInfo objects.
    """
    return send_ssdp_search(mx_seconds)


def print_device_info(device: DeviceInfo) -> str:
    """Return a formatted string with device information."""
    lines = [
        f"  Friendly Name: {device.friendly_name}",
        f"  Model: {device.model_name}",
        f"  Manufacturer: {device.manufacturer}",
        f"  UUID: {device.uuid}",
        f"  Device Type: {device.device_type}",
        f"  Location: {device.location_url}",
        f"  Source: {device.source_ip}:{device.source_port}",
    ]
    if device.services:
        lines.append(f"  Services: {', '.join(device.services)}")
    if device.control_urls:
        lines.append("  Control URLs:")
        for st, url in device.control_urls.items():
            lines.append(f"    {st}: {url}")
    return "\n".join(lines)

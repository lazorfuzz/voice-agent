"""One-time: fetch the Roomba's LOCAL control password from the iRobot cloud (needed because
newer firmware disabled on-robot retrieval). After this, control is 100% local via MQTT:8883.
Usage: roomba_cloud_pw.py <irobot_email> <irobot_password>  -> saves blid+password to /tmp."""
import json
import sys
import urllib.parse
import urllib.request


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=20))


def _post_form(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode())
    return json.load(urllib.request.urlopen(req, timeout=20))


def _post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def main(email, password):
    # 1) discovery -> gigya key + iRobot httpBase
    disc = _get("https://disc-prod.iot.irobotapi.com/v1/discover/endpoints?country_code=US")
    deployment = disc["deployments"][next(iter(disc["deployments"]))]
    http_base = deployment["httpBase"]
    apikey = disc["gigya"]["api_key"]
    gigya_base = disc["gigya"]["datacenter_domain"]

    # 2) gigya login -> UID + signature
    try:
        g = _post_form(f"https://accounts.{gigya_base}/accounts.login", {
            "apiKey": apikey, "targetenv": "mobile", "loginID": email,
            "password": password, "format": "json", "targetEnv": "mobile",
        })
    except urllib.error.HTTPError as e:
        print("gigya login HTTP error:", e.read().decode("utf-8", "replace")[:300])
        return 1
    if "UID" not in g:
        print("gigya login failed:", g.get("errorMessage") or g.get("statusReason") or g)
        return 1

    # 3) iRobot login -> robots{blid: {password, name, ...}}
    login = _post_json(f"{http_base}/v2/login", {
        "app_id": "ANDROID-C7FB240E-DF34-42D7-AE4E-A8C17079A294",
        "assume_robot_ownership": "0",
        "gigya": {"signature": g["UIDSignature"], "timestamp": g["signatureTimestamp"], "uid": g["UID"]},
    })
    robots = login.get("robots", {})
    if not robots:
        print("no robots on this iRobot account:", json.dumps(login)[:300])
        return 1
    for blid, info in robots.items():
        pw = info.get("password", "")
        print(f"ROBOT  name={info.get('name')!r}  blid={blid}  sku={info.get('sku')}  "
              f"password_len={len(pw)}")
        open("/tmp/roomba_blid.txt", "w").write(blid)
        open("/tmp/roomba_pw.txt", "w").write(pw)
    print("saved blid -> /tmp/roomba_blid.txt, password -> /tmp/roomba_pw.txt")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: roomba_cloud_pw.py <email> <password>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

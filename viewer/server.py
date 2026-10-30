#!/usr/bin/env python3
"""ygo-viewer: serves the replay board on the game host :8080.
  /                 the viewer page (index.html)
  /api/games        replay list, newest first: file, name, seats, winner, reason, turns, steps, finished
  /replays/<file>   one replay JSON (written by ygo-runner)
  /api/live         games being played right now (proxied from the game server's /live)
Stdlib only; runs as the systemd unit ygo-viewer."""
import json
import os
import sys
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAYS = Path(os.environ.get("YGO_REPLAYS", "/srv/ygo/replays"))
PORT = int(os.environ.get("YGO_VIEWER_PORT", "8080"))
_cache = {}  # file -> (mtime, summary)


def summary(p: Path):
    st = p.stat()
    hit = _cache.get(p.name)
    if hit and hit[0] == st.st_mtime:
        return hit[1]
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    s = {"file": p.name, "name": d.get("name"), "seats": d.get("seats"), "winner": d.get("winner"),
         "reason": d.get("reason"), "turns": d.get("turns"), "lp": d.get("lp"), "steps": len(d.get("steps", [])),
         "finished": d.get("finished") or int(st.st_mtime), "bytes": st.st_size}
    _cache[p.name] = (st.st_mtime, s)
    return s


class H(BaseHTTPRequestHandler):
    def send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/games":
            files = list(REPLAYS.glob("*.json"))
            games = [s for p in files if (s := summary(p))]
            live = {p.name for p in files}
            for k in [k for k in list(_cache) if k not in live]:
                _cache.pop(k, None)  # replay deleted: forget it, so the cache never outgrows the directory
            games.sort(key=lambda s: (s["finished"], s["file"]), reverse=True)
            return self.send(200, json.dumps(games).encode(), "application/json")
        if path.startswith("/replays/"):
            name = path[len("/replays/"):]
            p = REPLAYS / name
            if "/" in name or not name.endswith(".json") or not p.is_file():
                return self.send(404, b"not found", "text/plain")
            return self.send(200, p.read_bytes(), "application/json")
        if path == "/api/live":
            try:
                with urllib.request.urlopen(os.environ.get("YGO_SERVER", "http://127.0.0.1:8765") + "/live", timeout=5) as r:
                    return self.send(200, r.read(), "application/json")
            except OSError:
                return self.send(200, b"[]", "application/json")  # game server restarting: nothing live
        if path == "/health":
            return self.send(200, b"ok", "text/plain")
        self.send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    print(f"ygo-viewer on :{PORT}, replays in {REPLAYS}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

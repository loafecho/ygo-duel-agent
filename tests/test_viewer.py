"""viewer/server.py over a real socket: replay listing, replay files, path checks and the live fallback."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from pytest import fixture

from conftest import load


@fixture
def viewer(tmp_path, monkeypatch):
    replays = tmp_path / "replays"
    replays.mkdir()
    old = {"name": "old", "seats": ["model", "random"], "winner": 0, "reason": "lp", "turns": 9,
           "steps": [{}, {}], "finished": 100}
    new = {"name": "new", "seats": ["random", "model"], "winner": 1, "reason": "deck", "turns": 20,
           "steps": [{}], "finished": 200}
    (replays / "a.json").write_text(json.dumps(old))
    (replays / "b.json").write_text(json.dumps(new))
    (replays / "broken.json").write_text("{not json")
    (tmp_path / "secret.json").write_text("{}")
    monkeypatch.setenv("YGO_REPLAYS", str(replays))
    monkeypatch.setenv("YGO_SERVER", "http://127.0.0.1:9")  # nothing listens: /api/live must degrade to []
    mod = load("ygo_viewer", "viewer/server.py")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.H)
    mod.H.log_message = lambda *a: None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_games_newest_first_and_broken_files_skipped(viewer):
    code, body = get(viewer + "/api/games")
    games = json.loads(body)
    assert code == 200
    assert [g["name"] for g in games] == ["new", "old"]
    assert games[1]["steps"] == 2 and games[1]["seats"] == ["model", "random"]


def test_replay_file_and_path_checks(viewer):
    code, body = get(viewer + "/replays/a.json")
    assert code == 200 and json.loads(body)["name"] == "old"
    assert get(viewer + "/replays/missing.json")[0] == 404
    assert get(viewer + "/replays/a.txt")[0] == 404
    assert get(viewer + "/replays/../secret.json")[0] == 404
    assert get(viewer + "/replays/..%2fsecret.json")[0] == 404


def test_index_health_and_live_fallback(viewer):
    code, body = get(viewer + "/")
    assert code == 200 and b"Duel Replays" in body
    assert get(viewer + "/health") == (200, b"ok")
    assert get(viewer + "/api/live") == (200, b"[]")
    assert get(viewer + "/nope")[0] == 404

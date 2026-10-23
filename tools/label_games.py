#!/usr/bin/env python3
"""WindBot vs WindBot games on a game server whose seats log imitation labels ("windbot-labels" seats write one
JSON line per decision with 2+ options to the server's YGO_LABELS file on the game host). This only queues the games and
collects the results; the labels themselves land on the game host.

  python3 tools/label_games.py --server http://198.51.100.10:8766 --games 200
Only one consumer may poll /next at a time (results are handed out once).
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://198.51.100.10:8766")
ap.add_argument("--games", type=int, default=200)
a = ap.parse_args()


def call(path, body=None, timeout=60):
    req = urllib.request.Request(a.server + path, None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


games = [{"seats": ["windbot-labels", "windbot-labels"]} for _ in range(a.games)]
job = call("/job", {"name": "windbot-labels", "seed": int(time.time()), "games": games})["job"]
print(json.dumps({"event": "job", "job": job, "games": a.games}), flush=True)
results, t0 = [], time.time()
while len(results) < a.games:
    nxt = call("/next?max=64&wait=5&gather=50")
    if nxt["requests"]:
        call("/answer", [{"id": r["id"], "choice": 0} for r in nxt["requests"]])  # none expected: no remote seats
    for res in nxt["results"]:
        if res.get("job") == job:
            results.append(res)
            print(json.dumps({"event": "game", "idx": res["idx"], "winner": res.get("winner"), "turns": res.get("turns"),
                              "error": res.get("error"), "windbot": res.get("windbot")}), flush=True)
errors = sum("error" in r for r in results)
fallbacks = sum(w.get("fallbacks", 0) for r in results for w in r.get("windbot") or [])
print(f"label_games={len(results)} errors={errors} windbot_fallbacks={fallbacks} minutes={(time.time() - t0) / 60:.1f}")

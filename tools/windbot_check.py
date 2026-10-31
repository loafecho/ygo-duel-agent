#!/usr/bin/env python3
"""WindBot vs random through the game server: posts one job (WindBot in seat 0 for half the games, seat 1
for the rest), collects its results from /next and prints windbot_games=N errors=E windbot_win_rate=X.
Run on the game host; /next returns this job's results only to the holder of its token."""
import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8765"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
replays = int(sys.argv[2]) if len(sys.argv) > 2 else 2

TOKEN = ""
AUTH = {"Authorization": "Bearer " + open(os.environ["YGO_BROKER_AUTH_FILE"]).read().strip()} if os.environ.get("YGO_BROKER_AUTH_FILE") else {}  # the game server's shared secret, if it has one

def call(path, body=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST",
                                 headers={"X-Job-Token": TOKEN, **AUTH})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

games = [{"seats": ["windbot", "random"] if i % 2 == 0 else ["random", "windbot"], "replay": i < replays} for i in range(n)]
posted = call("/job", {"name": "windbot-check", "seed": int(time.time()), "games": games})
job, TOKEN = posted["job"], posted["token"]
results, t0 = [], time.time()
while len(results) < n and time.time() - t0 < 3600:
    r = call("/next?max=0&wait=10")
    results += [x for x in r.get("results", []) if x.get("job") == job]
errors = wins = fallbacks = decisions = 0
for x in results:
    if "error" in x:
        errors += 1
        print("error:", x["idx"], x["error"])
        continue
    seat = x["seats"].index("windbot")
    wins += x["winner"] == seat
    for w in x.get("windbot", []):
        fallbacks += w["fallbacks"]; decisions += w["decisions"]
    print(json.dumps({k: x.get(k) for k in ("idx", "seats", "winner", "reason", "turns", "seconds", "windbot", "replay")}))
done = len(results)
print(f"windbot_decisions={decisions} windbot_fallbacks={fallbacks} seconds={time.time() - t0:.0f}")
print(f"windbot_games={done} errors={errors} windbot_win_rate={wins / max(1, done - errors):.2f}")

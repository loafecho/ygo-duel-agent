#!/usr/bin/env python3
"""WindBot vs WindBot games on a game server whose seats log imitation labels ("windbot-labels" seats write one
JSON line per decision with 2+ options to the server's YGO_LABELS file on the game host). This only queues the games and
collects the results; the labels themselves land on the game host.

  python3 tools/label_games.py --server http://198.51.100.10:8766 --games 200
/next returns this job's results only to the holder of the token POST /job hands back.
"""
import argparse, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://198.51.100.10:8766")
ap.add_argument("--games", type=int, default=200)
# round 5: the labelled seat plays --deck (a WindBot deck name); its opponent is a plain WindBot playing --opp-decks in
# turn (both seat orders each); without them, round 4's Cyber Dragon mirror with both seats labelled
ap.add_argument("--deck", default="")
ap.add_argument("--opp-decks", default="")
a = ap.parse_args()


TOKEN = ""
AUTH = {"Authorization": "Bearer " + open(os.environ["YGO_BROKER_AUTH_FILE"]).read().strip()} if os.environ.get("YGO_BROKER_AUTH_FILE") else {}  # the game server's shared secret, if it has one


def call(path, body=None, timeout=60):
    req = urllib.request.Request(a.server + path, None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-Job-Token": TOKEN, **AUTH})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


if a.deck or a.opp_decks:
    opps = [d or None for d in a.opp_decks.split(",")] if a.opp_decks else [a.deck or None]
    games = []
    for g in range(a.games):
        o = opps[(g // 2) % len(opps)]
        games.append({"seats": ["windbot-labels", "windbot"] if g % 2 == 0 else ["windbot", "windbot-labels"],
                      "decks": [a.deck or None, o] if g % 2 == 0 else [o, a.deck or None]})
else:
    games = [{"seats": ["windbot-labels", "windbot-labels"]} for _ in range(a.games)]
posted = call("/job", {"name": "windbot-labels", "seed": int(time.time()), "games": games})
job, TOKEN = posted["job"], posted["token"]
print(json.dumps({"event": "job", "job": job, "games": a.games}), flush=True)
results, t0 = [], time.time()
while len(results) < a.games:
    nxt = call("/next?max=64&wait=5&gather=50")
    if nxt["requests"]:
        call("/answer", [{"id": r["id"], "lease": r["lease"], "choice": 0} for r in nxt["requests"]])  # none expected: no remote seats
    for res in nxt["results"]:
        if res.get("job") == job:
            results.append(res)
            print(json.dumps({"event": "game", "idx": res["idx"], "seed": res.get("seed"), "winner": res.get("winner"), "turns": res.get("turns"),
                              "error": res.get("error"), "windbot": res.get("windbot")}), flush=True)
errors = sum("error" in r for r in results)
fallbacks = sum(w.get("fallbacks", 0) for r in results for w in r.get("windbot") or [])
print(f"label_games={len(results)} errors={errors} windbot_fallbacks={fallbacks} minutes={(time.time() - t0) / 60:.1f}")

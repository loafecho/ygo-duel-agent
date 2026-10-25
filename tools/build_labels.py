#!/usr/bin/env python3
"""Round 3's imitation label set: WindBot's own picks (windbot-labels.jsonl from the game host, one line per WindBot decision
with 2+ options) plus a teacher's picks (a teacher seat's pick log, one JSON line per decision).
Output: one JSON line per decision, {"src", "prompt", "n", "choice", "held"}, the prompt in the runner's format.
Held-out: every WindBot game whose seed hashes to 0 mod 10 (whole games, so no position leaks into training);
teacher labels are all training data.

  python3 tools/build_labels.py --windbot runs/windbot-labels.jsonl --teacher runs/teacher-picks.jsonl \\
      --teacher runs/kev4b-picks.jsonl:1792766426001 --out runs/labels-r3.jsonl
A teacher file may carry ":<job>" to keep only that job's picks.
"""
import argparse, hashlib, json, random

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ap = argparse.ArgumentParser()
ap.add_argument("--windbot", required=True)
ap.add_argument("--teacher", action="append", default=[])
ap.add_argument("--out", required=True)
# round 4: games come from label_games.py runs (their "game" lines carry seed and winner)
ap.add_argument("--results", action="append", default=[], help="label_games.py output; only these games' labels are kept")
ap.add_argument("--sl-job", default="", help="round 4: only this job's games go into --out (the rest feed --value-out)")
ap.add_argument("--value-out", default="", help="round 4: critic data, decisions with the game's outcome")
ap.add_argument("--value-per-game", type=int, default=40)
a = ap.parse_args()


def prompt(state, cards, kind, options, history=""):  # board::prompt in runner/src/board.rs (round 4: history)
    return f"{state}{history}{cards}Decision: {kind}.\nOptions:\n" + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options)) + "Answer:"


games = {}  # seed -> (job, winner)
for path in a.results:
    job = None
    for line in open(path):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("event") == "job":
            job = str(r["job"])
        elif r.get("event") == "game" and r.get("seed") is not None and not r.get("error"):
            games[r["seed"]] = (job, r["winner"])


def held(seed):
    return int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % 10 == 0


rows, counts, value = [], {}, {}
with open(a.windbot) as f:
    for line in f:
        r = json.loads(line)
        if len(r["options"]) < 2 or not 0 <= r["choice"] < len(r["options"]):
            continue
        p = prompt(r["state"], r["cards"], r["kind"], r["options"], r.get("history", ""))
        if a.results:
            if r["seed"] not in games:
                continue  # a game from a run whose results were not kept
            job, winner = games[r["seed"]]
            if job != a.sl_job:
                value.setdefault(r["seed"], []).append({"prompt": p, "n": len(r["options"]), "seed": r["seed"],
                                                        "player": r["player"], "won": int(winner == r["player"]),
                                                        "held": held(r["seed"])})
                continue
        rows.append({"src": "windbot", "prompt": p, "n": len(r["options"]), "choice": r["choice"], "held": held(r["seed"])})
for spec in a.teacher:
    path, _, job = spec.partition(":")
    src = "teacher" if "teacher" in path else path.rsplit("/", 1)[-1].split("-")[0]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if job and str(r["job"]) != job:
                continue
            if r["n"] < 2 or not 0 <= r["choice"] < r["n"]:
                continue
            rows.append({"src": src, "prompt": r["prompt"], "n": r["n"], "choice": r["choice"], "held": False})
random.Random(1).shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        k = (r["src"], r["held"])
        counts[k] = counts.get(k, 0) + 1
if a.value_out:  # an even spread over each game: up to value-per-game decisions, both seats
    vrows = []
    for seed, ds in value.items():
        step = max(1, len(ds) // a.value_per_game)
        vrows += ds[::step][: a.value_per_game]
    with open(a.value_out, "w") as f:
        for r in vrows:
            f.write(json.dumps(r) + "\n")
    print(f"value games={len(value)} decisions={len(vrows)} held_games={sum(held(s) for s in value)} file={a.value_out}")
wb = sum(v for (s, _), v in counts.items() if s == "windbot")
teacher = sum(v for (s, _), v in counts.items() if s != "windbot")
print(json.dumps({f"{s}{'_held' if h else ''}": v for (s, h), v in sorted(counts.items())}))
print(f"labels windbot={wb} teacher={teacher} windbot_heldout={counts.get(('windbot', True), 0)} file={a.out}")

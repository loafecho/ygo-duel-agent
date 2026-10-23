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
a = ap.parse_args()


def prompt(state, cards, kind, options):  # board::prompt in runner/src/board.rs
    return f"{state}{cards}Decision: {kind}.\nOptions:\n" + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options)) + "Answer:"


def held(seed):
    return int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % 10 == 0


rows, counts = [], {}
with open(a.windbot) as f:
    for line in f:
        r = json.loads(line)
        if len(r["options"]) < 2 or not 0 <= r["choice"] < len(r["options"]):
            continue
        rows.append({"src": "windbot", "prompt": prompt(r["state"], r["cards"], r["kind"], r["options"]),
                     "n": len(r["options"]), "choice": r["choice"], "held": held(r["seed"])})
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
wb = sum(v for (s, _), v in counts.items() if s == "windbot")
teacher = sum(v for (s, _), v in counts.items() if s != "windbot")
print(json.dumps({f"{s}{'_held' if h else ''}": v for (s, h), v in sorted(counts.items())}))
print(f"labels windbot={wb} teacher={teacher} windbot_heldout={counts.get(('windbot', True), 0)} file={a.out}")

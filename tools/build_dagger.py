#!/usr/bin/env python3
"""Round 5 DAgger aggregation: an existing label set (build_labels.py format: {"src","prompt","n","choice","held"})
plus the shadow-WindBot labels of one collection run (the runner's YGO_DAGGER_LABELS lines), kept only for the games
of that run (their seeds, from the worker's match-state.json). Held-out: seed hash 0 mod 10, as build_labels.py.

  python3 tools/build_dagger.py --base labels-dagger.jsonl --dagger dagger-labels.jsonl \\
      --state m-dagger2-collect/match-state.json --src dagger2 --out labels-dagger2.jsonl --base-ratio 1
Round 5: --base-ratio 1 trains on the new corrections plus as many old labels (warm-started models keep the rest).
"""
import argparse, hashlib, json, random

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--dagger", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--src", default="dagger2")
ap.add_argument("--out", required=True)
ap.add_argument("--base-ratio", type=float, default=0, help="keep only this many base training rows per new label "
                "(a random sample; every held-out row is kept for the held-out check); 0 = keep all")
a = ap.parse_args()


def prompt(state, cards, kind, options, history=""):  # board::prompt in runner/src/board.rs
    return f"{state}{history}{cards}Decision: {kind}.\nOptions:\n" + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options)) + "Answer:"


def held(seed):
    return int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % 10 == 0


seeds = {r["seed"] for r in json.load(open(a.state))["results"].values() if "error" not in r}
rows = [json.loads(l) for l in open(a.base)]
base_n = len(rows)
new = 0
for line in open(a.dagger):
    r = json.loads(line)
    if r["seed"] not in seeds or len(r["options"]) < 2 or not 0 <= r["choice"] < len(r["options"]):
        continue
    rows.append({"src": a.src, "prompt": prompt(r["state"], r["cards"], r["kind"], r["options"], r.get("history", "")),
                 "n": len(r["options"]), "choice": r["choice"], "held": held(r["seed"])})
    new += 1
if a.base_ratio:
    base, extra = rows[:base_n], rows[base_n:]
    train = [r for r in base if not r["held"]]
    random.Random(2).shuffle(train)
    rows = train[: int(a.base_ratio * new)] + [r for r in base if r["held"]] + extra
    base_n = len(rows) - len(extra)
random.Random(1).shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"dagger_labels base={base_n} (ratio {a.base_ratio or 'all'}) new={new} games={len(seeds)} total={len(rows)} file={a.out}")

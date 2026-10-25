#!/usr/bin/env python3
"""Round 4: every distinct card of the deck (main + extra) must have a one-line summary, keyed by its CDB name
(the worker raises on a missing one instead of falling back to the first sentence). Run where the CDB is (the game host).

  python3 tools/oneliners_check.py --deck /srv/ygo/AI_CyberDragon.ydk --cdb /srv/ygo/cdb/cards.cdb \\
      --file worker/oneliners-r4.json
Prints "oneliners cards=N missing=M" (and each missing name); exit 1 if any is missing.
"""
import argparse, json, sqlite3, sys

ap = argparse.ArgumentParser()
ap.add_argument("--deck", default="/srv/ygo/AI_CyberDragon.ydk")
ap.add_argument("--cdb", default="/srv/ygo/cdb/cards.cdb")
ap.add_argument("--file", required=True)
a = ap.parse_args()

codes = []
for line in open(a.deck):
    line = line.strip()
    if line.isdigit() and int(line) not in codes:
        codes.append(int(line))
db = sqlite3.connect(a.cdb)
names = []
for c in codes:
    row = db.execute("select name from texts where id=?", (c,)).fetchone()
    names.append(row[0] if row else f"<no CDB name for {c}>")
summaries = json.load(open(a.file))
missing = [n for n in dict.fromkeys(names) if not summaries.get(n, "").strip()]
extra = sorted(set(summaries) - set(names))
for n in missing:
    print(f"missing: {n}")
if extra:
    print(f"not in the deck (ignored): {extra}")
print(f"oneliners cards={len(dict.fromkeys(names))} missing={len(missing)}")
sys.exit(1 if missing else 0)

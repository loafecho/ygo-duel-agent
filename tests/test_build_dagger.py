"""tools/build_dagger.py keeps only label rows from the collection run's own games, once each."""
import json
import subprocess
import sys

from conftest import ROOT


def row(job, game, decision, seed=53001, **kw):
    r = {"seed": seed, "job": job, "game": game, "decision": decision, "state": "s", "history": "", "cards": "",
         "kind": "k", "options": ["a", "b"], "choice": 1}
    r.update(kw)
    return r


def test_rows_must_come_from_the_runs_own_jobs(tmp_path):
    state = {"jobs": {"100": [0, 1]}, "results": {"0": {"job": 100, "idx": 0, "seed": 53001},
                                                    "1": {"job": 100, "idx": 1, "seed": 53002, "error": "x"}}}
    rows = [row(100, 0, 1), row(100, 0, 1),  # duplicate of the same decision
            row(100, 0, 2),
            row(999, 0, 1),  # another job on the same seed
            row(100, 1, 1, seed=53002),  # a game that errored
            {k: v for k, v in row(100, 0, 3).items() if k != "job"}]  # no provenance
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "dagger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "base.jsonl").write_text("")
    out = subprocess.run([sys.executable, "-I", str(ROOT / "tools/build_dagger.py"), "--base", str(tmp_path / "base.jsonl"),
                          "--dagger", str(tmp_path / "dagger.jsonl"), "--state", str(tmp_path / "state.json"),
                          "--out", str(tmp_path / "out.jsonl")], capture_output=True, text=True, check=True).stdout
    assert "new=2 dropped=4 games=1" in out
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 2

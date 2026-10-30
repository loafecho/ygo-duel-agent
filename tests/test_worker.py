"""Pure helpers in worker/worker.py: prompt rewriting, match bookkeeping and the statistics behind the evals."""
import json
import math


PROMPT = (
    "Turn 3, my main phase. Life points: me 6000, opponent 8000\n"
    "Recent plays (oldest first):\n"
    "- T1 opponent: Normal Summoned Card B\n"
    "- T2 me: activated Card A\n"
    "Card text:\n"
    "- Card A: Draw 2 cards. Then discard 1 card.\n"
    "- Card B: Once per turn: destroy 1 monster.\n"
    "Decision: choose an option\n"
    "A. Activate Card A\nB. Pass\nAnswer:"
)


def test_strip_history_drops_only_the_recent_plays_lines(worker):
    out = worker.strip_history(PROMPT)
    assert "Recent plays" not in out and "- T1 " not in out and "- T2 " not in out
    assert out.startswith("Turn 3, my main phase.")
    assert "Card text:\n- Card A:" in out
    assert worker.strip_history("no history here") == "no history here"


def test_render_cards_full_keeps_the_prompt(worker):
    assert worker.render_cards(PROMPT, mode="full", history=True) == PROMPT


def test_render_cards_none_removes_the_card_text_section(worker):
    out = worker.render_cards(PROMPT, mode="none", history=True)
    assert "Card text:" not in out and "Draw 2 cards" not in out
    assert "Decision: choose an option" in out and out.endswith("Answer:")


def test_render_cards_oneliner_uses_the_summary_file(worker, tmp_path):
    table = tmp_path / "summaries.json"
    table.write_text(json.dumps({"Card A": "draw 2, discard 1"}))
    out = worker.render_cards(PROMPT, mode="oneliner", oneliners=str(table), history=True)
    assert "- Card A: draw 2, discard 1" in out
    # a card missing from the file falls back to a marked first sentence instead of failing
    assert "- Card B: (no summary) Once per turn: destroy 1 monster" in out


def test_render_cards_history_flag(worker):
    out = worker.render_cards(PROMPT, mode="full", history=False)
    assert "Recent plays" not in out and "Card text:" in out


def test_seats_vs_alternates_who_goes_first(worker):
    games = worker.seats_vs("model", "windbot", 4)
    assert [g["seats"] for g in games] == [["model", "windbot"], ["windbot", "model"]] * 2
    assert all("decks" not in g for g in games)


def test_seats_vs_rotates_pool_decks_in_both_seat_orders(worker):
    games = worker.seats_vs("model", "windbot", 4, opp_decks=["Blackwing", "Zoodiac"])
    assert [g["decks"] for g in games] == [[None, "Blackwing"], ["Blackwing", None], [None, "Zoodiac"], ["Zoodiac", None]]


def test_paired_vs_shares_a_seed_per_pair(worker):
    games = worker.paired_vs("a", "b", 4, seed=11)
    assert games[0]["seed"] == games[1]["seed"] != games[2]["seed"] == games[3]["seed"]
    assert games[0]["seats"] == ["a", "b"] and games[1]["seats"] == ["b", "a"]


def test_summarize_scores_from_the_measured_seat(worker):
    st = {
        "name": "m",
        "games": [{"seats": ["model", "windbot"]}, {"seats": ["windbot", "model"]}] * 2,
        "results": {
            "0": {"seats": ["model", "windbot"], "winner": 0, "turns": 10},
            "1": {"seats": ["windbot", "model"], "winner": 1, "turns": 12},
            "2": {"seats": ["model", "windbot"], "winner": -1, "turns": 40},
            "3": {"error": "engine stalled"},
        },
    }
    s = worker.summarize(st)
    assert (s["wins"], s["losses"], s["draws"], s["errors"], s["games"]) == (2, 0, 1, 1, 4)
    assert s["win_rate"] == 0.5  # draws and errors count against the win rate
    assert s["mean_turns"] == round((10 + 12 + 40) / 3, 1)
    assert s["errors_detail"] == ["engine stalled"]


def test_wilson_interval(worker):
    assert worker.wilson(0, 0) == (0.0, 1.0)
    lo, hi = worker.wilson(468, 500)
    assert lo < 468 / 500 < hi
    assert (lo, hi) == (0.911, 0.9543)  # the interval reported for 468/500 wins in deploy/hf/README.md


def test_auc(worker):
    assert worker.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert worker.auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert worker.auc([0.5, 0.5], [0, 1]) == 0.5  # ties share a rank


def test_lp_phi(worker):
    assert math.isclose(worker.lp_phi("Life points: me 6000, opponent 8000"), -0.25)
    assert worker.lp_phi("no life points line") == 0.0


def test_held_game_is_stable_and_about_one_in_ten(worker):
    held = [worker.held_game(k) for k in range(2000)]
    assert held == [worker.held_game(k) for k in range(2000)]
    assert 150 < sum(held) < 250


def test_job_token_is_private_and_stable(worker):
    t = worker.job_token()
    p = worker.DATA / "broker-token"
    assert len(t) >= 32 and p.read_text() == t and (p.stat().st_mode & 0o777) == 0o600
    worker._JOB_TOKEN.clear()
    assert worker.job_token() == t


def test_job_token_replaces_empty_or_partial_file(worker):
    p = worker.DATA / "broker-token"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    worker._JOB_TOKEN.clear()
    t = worker.job_token()
    assert len(t) >= 32 and p.read_text() == t and (p.stat().st_mode & 0o777) == 0o600


def test_send_answers_reports_rejected_ids(worker, monkeypatch):
    import io
    import urllib.error

    def refuse(path, body=None, timeout=40):
        raise urllib.error.HTTPError(path, 409, "Conflict", {}, io.BytesIO(b'{"accepted": 1, "rejected": [7]}'))

    monkeypatch.setattr(worker, "http", refuse)
    assert worker.send_answers([{"id": 7, "lease": "x", "choice": 0}]) == [7]
    monkeypatch.setattr(worker, "http", lambda *a, **k: {"accepted": 1})
    assert worker.send_answers([{"id": 8, "lease": "y", "choice": 0}]) == []

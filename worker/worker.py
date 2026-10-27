"""ygo-worker: a plain venv process on the GPU workstation's 3060 (~/.local/share/ygo-worker/venv), started by supervisor.py.
It connects OUT to the game server on the game host (ygo-runner serve, :8765). Nothing listens on the GPU workstation.

Every decision arrives as a prompt (board, decision, lettered options, "Answer:") built by the runner.
A model scores it with its first-token logits over the option letters: one forward pass per decision.

Modes:
  train  self-play PPO on Qwen2.5-0.5B + LoRA (round 2: warm start from YGO_WARM, a pool .pt). Opponents: WindBot,
         itself, a pool of its last 5 checkpoints, and random (shares in recipe()).
         Reward +1 win, -1 loss or draw, plus LP-difference shaping that fades to 0 by half the GPU-hour cap, minus
         LOOP_PENALTY on a seat's decisions in a turn where the runner's loop breaker took over from it.
         Greedy evals (200 games vs WindBot, 200 vs random) run at the start and every EVAL_EVERY GPU-hours;
         training stops when vs WindBot reaches TARGET or the cap is spent ("train_stop reason=target|cap").
  zeroshot  the ZeroShot baseline: Qwen3.5-4B, no training, 200 games vs random.
  table  the final checkpoint and up to 4 earlier ones, 100 games each vs WindBot, ZeroShot, random, the previous one
         (round 3: YGO_TABLE_OPPS=windbot,ref, the reference being YGO_TABLE_REF, e.g. step-1239).
  textcheck  round 3, no training: YGO_WARM scored on held-out WindBot-labeled decisions (YGO_LABELS_FILE) with and
         without the card-text section; top-1 agreement with WindBot's pick decides the card-text form.

Round 3 (imitation first): with YGO_LABELS_FILE set, train starts with supervised cross-entropy on the labels
(WindBot's and the teachers' picks, the held-out games excluded), then the PPO loop above as a polish.
YGO_CARD_TEXT=full|oneliner|none rewrites the card-text section of every prompt the model scores.

SIGTERM (the supervisor's game guard) makes every mode save its state and exit so the GPU is freed;
the next start resumes. Games on the game host just wait for answers meanwhile.
Logs: one JSON object per line on stdout and <YGO_DATA>/logs/<mode>.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

MODEL = os.environ.get("YGO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ZEROSHOT_MODEL = os.environ.get("YGO_ZEROSHOT_MODEL", "Qwen/Qwen3.5-4B")
SERVER = os.environ.get("YGO_BROKER", "http://198.51.100.10:8765")
DATA = Path(os.environ.get("YGO_DATA", Path.home() / ".local/share/ygo-worker/data"))
CKPT = DATA / "ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# training recipe (YGO_RECIPE=2 is the one allowed change for a second run; see recipe())
CAP_H = float(os.environ.get("YGO_CAP_H", "24"))
TARGET = 0.50          # vs WindBot
LOOP_PENALTY = float(os.environ.get("YGO_LOOP_PENALTY", "0.1"))
WARM = os.environ.get("YGO_WARM", "")
EVAL_EVERY_H = float(os.environ.get("YGO_EVAL_EVERY_H", "4.0"))  # round 2: a WindBot+random eval costs ~1 GPU-hour
EVAL_GAMES = int(os.environ.get("YGO_EVAL_GAMES", "200"))
CKPT_EVERY_S = int(os.environ.get("YGO_POOL_EVERY_S", "1800"))  # a pool snapshot (opponent + table candidate); 35 MB each
SAVE_EVERY_S = 600     # the resumable checkpoint, so a crash costs at most this much
POOL = 5
BATCH = int(os.environ.get("YGO_BATCH", "32"))
MICRO = int(os.environ.get("YGO_MICRO", "4"))
BUFFER_CAP = 4096
KEEP_P = float(os.environ.get("YGO_KEEP_P", "1.0"))         # share of a finished game's decisions kept (was 0.35 in run 1: the buffer starved, so keep all)
PPO_EPOCHS = int(os.environ.get("YGO_PPO_EPOCHS", "2"))  # each decision is used for this many updates
LIVE_GAMES = int(os.environ.get("YGO_LIVE_GAMES", "96"))
MAX_TOKENS = 2048      # round 2 prompts carry card text (~600-900 tokens, more late in a game)
CARD_TEXT = os.environ.get("YGO_CARD_TEXT", "full")  # round 3: full | oneliner | none (set by the textcheck)
ONELINERS_FILE = os.environ.get("YGO_ONELINERS", "oneliners.json")  # round 4: oneliners-r4.json (checked vs the CDB)
LABELS_FILE = os.environ.get("YGO_LABELS_FILE", "")
SL_EPOCHS = float(os.environ.get("YGO_SL_EPOCHS", "1"))
SL_LR = float(os.environ.get("YGO_SL_LR", "1e-4"))
SL_CAP_H = float(os.environ.get("YGO_SL_CAP_H", "0"))  # round 5: imitation stops here (GPU-h) even mid-epoch
CAP2_H = float(os.environ.get("YGO_CAP2_H", "0"))  # round 3: at CAP_H, continue to this if the last eval beat the one before
EVAL_OPPS = os.environ.get("YGO_EVAL_OPPS", "windbot,random").split(",")
STOP = False
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
# round 5: per-seat decks (WindBot deck names; "" = the game server's --deck, Cyber Dragon)
MY_DECK = os.environ.get("YGO_DECK", "")
POOL_DECKS = [d for d in os.environ.get("YGO_POOL_DECKS", "Blackwing,Dragunity,Zoodiac,Trickstar,ThunderDragon").split(",") if d]
LORA_R = int(os.environ.get("YGO_LORA_R", "16"))


def recipe() -> dict:
    r = {"lr": 2e-5, "clip": 0.2, "entropy": 0.01, "shaping": 0.5, "lora_r": 16, "temp": 1.0, "random_share": 0.1,
         "windbot_share": 0.4, "pool_share": 0.25}
    if os.environ.get("YGO_RECIPE") in ("2", "3"):  # 3: round 3's polish shares
        r.update(json.loads(os.environ.get("YGO_RECIPE_CHANGE", "{}")))
    return r


def log(event: str, **kw) -> None:
    line = json.dumps({"t": round(time.time(), 3), "event": event, **kw})
    print(line, flush=True)
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    with open(DATA / "logs" / f"{MODE}.jsonl", "a") as f:
        f.write(line + "\n")


def on_term(signum, frame) -> None:
    global STOP
    STOP = True


def http(path: str, body=None, timeout: float = 40):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(SERVER + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw or b"null")
    except ValueError:
        return raw.decode(errors="replace")  # /answer and /health reply with plain "ok"


class Stopped(Exception):
    """SIGTERM arrived while waiting on the game server."""


def server_call(path: str, body=None, timeout: float = 40):
    """Retry until the game server answers (it may be restarting); gives up only on STOP."""
    while True:
        try:
            return http(path, body, timeout)
        except Exception as e:
            if STOP:
                raise Stopped() from e
            log("server_wait", path=path, err=f"{type(e).__name__}: {e}"[:200])
            time.sleep(3)


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


# ------------------------------------------------------------------ card text

_ONELINERS: dict[str, dict[str, str]] = {}
_MISSING: set[str] = set()
HISTORY = os.environ.get("YGO_HISTORY", "1") != "0"  # round 4: keep the runner's "Recent plays" section
HIST_HEAD = "Recent plays (oldest first):\n"
R3_FMT = {"oneliners": "oneliners.json", "history": False}  # how round-3 checkpoints saw their prompts


def strip_history(prompt: str) -> str:
    if HIST_HEAD not in prompt:
        return prompt
    head, _, rest = prompt.partition(HIST_HEAD)
    lines = rest.split("\n")
    k = 0
    while k < len(lines) and re.match(r"- T\d+ ", lines[k]):
        k += 1
    return head + "\n".join(lines[k:])


def render_cards(prompt: str, mode: str | None = None, oneliners: str | None = None, history: bool | None = None) -> str:
    """The prompt with its "Card text:" section kept (full), cut to one hand-written line per card (oneliner),
    or removed (none). Round 4: `history` False drops the "Recent plays" section; `oneliners` picks the file."""
    mode = mode or CARD_TEXT
    if not (HISTORY if history is None else history):
        prompt = strip_history(prompt)
    if mode == "full" or "Card text:\n" not in prompt:
        return prompt
    head, _, rest = prompt.partition("Card text:\n")
    cards, sep, tail = rest.partition("Decision: ")
    if mode == "none":
        return head + sep + tail
    fname = oneliners or ONELINERS_FILE
    if fname not in _ONELINERS:
        _ONELINERS[fname] = json.loads((Path(__file__).parent / fname).read_text())
    table = _ONELINERS[fname]
    lines = []
    for line in cards.splitlines():
        name, _, text = line.removeprefix("- ").partition(": ")
        if name not in table:  # round 4: no silent first-sentence fallback
            if os.environ.get("YGO_ONELINER_STRICT", "0") == "1":
                raise KeyError(f"no one-line summary for {name!r} in {fname}")
            # round 5: opponents' tokens and other off-list cards: a marked, logged first sentence instead of a crash
            if name not in _MISSING:
                _MISSING.add(name)
                log("oneliner_missing", card=name, file=fname)
            lines.append(f"- {name}: (no summary) {text.split('. ')[0][:200]}")
            continue
        lines.append(f"- {name}: {table[name]}")
    return head + "Card text:\n" + "\n".join(lines) + "\n" + sep + tail


def load_labels(held: bool | None = None, src: str | None = None) -> list[dict]:
    out = []
    with open(LABELS_FILE) as f:
        for line in f:
            r = json.loads(line)
            if (held is None or r["held"] == held) and (src is None or r["src"] == src):
                out.append(r)
    return out


# ------------------------------------------------------------------ scoring

class Scorer:
    """A causal LM that answers a lettered prompt with its next-token logits over the option letters."""

    def __init__(self, model_id: str, chat: bool, lora_r: int = 0, budget: int = 24000):
        from transformers import AutoTokenizer
        self.model_id, self.chat, self.budget = model_id, chat, budget
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        self.model = load_lm(model_id, dtype)
        if lora_r:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(self.model, cfg)
        self.model.eval()
        sample = self.format("Options:\nA. x\nB. y\nAnswer:")
        base = self.tok(sample, add_special_tokens=False).input_ids
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok(sample + self.letter_text(L), add_special_tokens=False).input_ids
            if ids[: len(base)] != base or len(ids) != len(base) + 1:
                raise RuntimeError(f"letter {L!r} is not one token after the prompt in {model_id}")
            self.letter_ids.append(ids[-1])
        self.letter_ids_t = torch.tensor(self.letter_ids, device=DEVICE)

    def letter_text(self, L: str) -> str:
        return L if self.chat else " " + L

    def format(self, prompt: str, fmt: dict | None = None) -> str:
        prompt = render_cards(prompt, **(fmt or {}))
        if not self.chat:
            return prompt
        body = prompt.removesuffix("Answer:") + "Reply with only the letter of the best option."
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def encode(self, prompts: list[str], fmt: dict | None = None) -> list[list[int]]:
        out = []
        for p in prompts:
            ids = self.tok(self.format(p, fmt), add_special_tokens=False).input_ids
            out.append(ids[-MAX_TOKENS:])  # keep the end: the options and "Answer:" must survive
        return out

    def feature_hook(self) -> None:
        """Round 4 critic: keep the final norm's output at the last position (the "Answer:" end, left padding) of
        every forward pass in self.feat, for a value head reading the board the policy reads."""
        norm = [m for n, m in self.model.named_modules() if n.endswith("model.norm")][-1]
        self.feat = None

        def hook(mod, inp, out):
            self.feat = out[:, -1, :]
        norm.register_forward_hook(hook)

    def logits(self, prompts: list[str], ns: list[int], grad: bool = False, fmt: dict | None = None,
               feats: list | None = None) -> list[torch.Tensor]:
        """Letter logits (float32, length n_i) for each prompt, in input order. With `feats` (a list) and the
        feature hook on, also fills feats[i] with prompt i's final hidden state (float32, grad kept if `grad`)."""
        enc = self.encode(prompts, fmt)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out: list = [None] * len(enc)
        if feats is not None:
            feats[:] = [None] * len(enc)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(enc[order[j]]) <= self.budget:
                j += 1
            idx = order[i:j]
            width = len(enc[idx[-1]])
            ids = torch.full((len(idx), width), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for r, k in enumerate(idx):
                ids[r, width - len(enc[k]):] = torch.tensor(enc[k])
                mask[r, width - len(enc[k]):] = 1
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            try:
                with torch.set_grad_enabled(grad), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    last = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
            except torch.OutOfMemoryError:
                if grad or len(idx) == 1:
                    raise
                del ids, mask
                torch.cuda.empty_cache()
                self.budget = max(width, self.budget // 2)  # smaller batches from now on; redo this one
                log("oom_shrink", budget=self.budget, batch=len(idx), width=width)
                continue
            sel = last.float()[:, self.letter_ids_t]
            for r, k in enumerate(idx):
                out[k] = sel[r, : ns[k]]
                if feats is not None:
                    feats[k] = self.feat[r].float()
            i = j
        return out


def load_lm(model_id: str, dtype):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        m, info = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, output_loading_info=True)
        if info["missing_keys"]:  # a text-only class over a multimodal checkpoint must not start from random weights
            raise ValueError(f"{len(info['missing_keys'])} missing weights, e.g. {info['missing_keys'][:3]}")
    except (ValueError, KeyError) as e:
        log("load_fallback", model=model_id, err=str(e)[:300])
        m = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)  # multimodal (Qwen3.5), text-only use
    return m.to(DEVICE)


def lora_params(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


class Weights:
    """Swap a stored LoRA state (a pool or table checkpoint) in and out of the live model."""

    def __init__(self, model, state: dict[str, torch.Tensor] | None):
        self.params, self.state = lora_params(model), state

    def __enter__(self):
        if self.state is not None:
            self.saved = {n: p.detach().clone() for n, p in self.params.items()}
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(self.state[n])  # every pool/table state holds the same LoRA names

    def __exit__(self, *a):
        if self.state is not None:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(self.saved[n])


def choose(sc: Scorer, reqs: list[dict], greedy: bool, temp: float = 1.0, state=None, fmt: dict | None = None):
    """(choice, logp of choice) per request. `fmt`: render options for this player (R3_FMT for round-3 models)."""
    if not reqs:
        return []
    with Weights(sc.model, state):
        lg = sc.logits([r["prompt"] for r in reqs], [r["n"] for r in reqs], fmt=fmt)
    out = []
    for l in lg:
        lp = torch.log_softmax(l / temp, -1)
        a = int(torch.argmax(lp)) if greedy else int(torch.multinomial(lp.exp(), 1))
        out.append((a, float(lp[a])))
    return out


# ------------------------------------------------------------------ matches (eval, zeroshot, table)

def seats_vs(tag: str, opp: str, games: int, opp_decks: list[str] | None = None) -> list[dict]:
    """Half the games with `tag` as player 0 (goes first), half as player 1. Round 5: `tag` plays MY_DECK; with
    `opp_decks` the opponent plays opp_decks[(g // 2) % len] (each pool deck in both seat orders)."""
    out = [{"seats": [tag, opp] if g % 2 == 0 else [opp, tag]} for g in range(games)]
    return with_decks(out, tag, opp_decks)


def with_decks(games: list[dict], tag: str, opp_decks: list[str] | None = None) -> list[dict]:
    if not MY_DECK and not opp_decks:
        return games
    for g, gm in enumerate(games):
        me = gm["seats"].index(tag)
        decks = [None, None]
        decks[me] = MY_DECK or None
        decks[1 - me] = opp_decks[(g // 2) % len(opp_decks)] if opp_decks else None
        gm["decks"] = decks
    return games


def run_match(name: str, games: list[dict], players: dict, state_file: Path, seed: int, replay_first: int = 2,
              on_tick=None) -> dict | None:
    """Submit a job (or resume the saved one), answer its requests with `players` (tag -> fn(reqs) -> choices),
    collect every result. Returns the summary, or None if stopped (state kept for the next start)."""
    st = json.loads(state_file.read_text()) if state_file.exists() else None
    if st is None or st.get("name") != name:
        for k, g in enumerate(games):
            g["replay"] = k < replay_first
            g["live"] = k < 2  # streamed to the viewer's Live page while it plays
        st = {"name": name, "jobs": {}, "games": games, "results": {}}
        submit_missing(st, seed)
        save_json(state_file, st)
        log("match_start", name=name, jobs=list(st["jobs"]), games=len(games))
    last_check = time.time()
    while len(st["results"]) < len(st["games"]):
        if STOP:
            return None
        nxt = server_call("/next?max=256&wait=2&gather=50", timeout=40)
        answers = []
        mine = [r for r in nxt["requests"] if str(r["job"]) in st["jobs"]]
        other = [r for r in nxt["requests"] if str(r["job"]) not in st["jobs"]]
        answers += [{"id": r["id"], "choice": random.randrange(r["n"])} for r in other]  # stale games: let them end
        by_tag: dict[str, list] = {}
        for r in mine:
            by_tag.setdefault(r["policy"], []).append(r)
        for tag, rs in by_tag.items():
            for r, (a, _) in zip(rs, players[tag](rs)):
                answers.append({"id": r["id"], "choice": a})
        if answers:
            server_call("/answer", answers)
        for res in nxt["results"]:
            idxs = st["jobs"].get(str(res["job"]))
            if idxs is not None:
                orig = idxs[res["idx"]]
                st["results"][str(orig)] = dict(res, idx=orig)
        if nxt["results"]:
            save_json(state_file, st)
        if time.time() - last_check > 30:  # games lost to a game-server restart are played again
            last_check = time.time()
            active = server_call("/status").get("active", {})
            if not any(j in active for j in st["jobs"]) and len(st["results"]) < len(st["games"]):
                log("match_resubmit", name=name, missing=len(st["games"]) - len(st["results"]))
                submit_missing(st, seed + len(st["jobs"]))
                save_json(state_file, st)
        if on_tick:
            on_tick()
    return summarize(st)


def submit_missing(st: dict, seed: int) -> None:
    missing = [i for i in range(len(st["games"])) if str(i) not in st["results"]]
    games = [dict(st["games"][i], live=k < 2) for k, i in enumerate(missing)]
    job = server_call("/job", {"name": st["name"], "seed": seed, "games": games})["job"]
    st["jobs"][str(job)] = missing


def summarize(st: dict) -> dict:
    """Score from the point of view of the tag in seat 0 of game 0 (the player being measured)."""
    tag = st["games"][0]["seats"][0]
    wins = losses = draws = errors = 0
    for res in st["results"].values():
        if "error" in res:
            errors += 1
            continue
        seat = res["seats"].index(tag)  # a shadow seat keeps its "shadow:" tag in seats
        if res["winner"] == seat:
            wins += 1
        elif res["winner"] == -1:
            draws += 1
        else:
            losses += 1
    n = len(st["results"])
    replays = [r["replay"] for r in st["results"].values() if r.get("replay")]
    loop_breaks = sum(r.get("loop_breaks", 0) for r in st["results"].values())
    shadow = [0, 0, 0]  # round 4 DAgger, the measured seat: asked, advice among the options, model chose otherwise
    for r in st["results"].values():
        if r.get("shadow") and tag in r["seats"]:
            for k in range(3):
                shadow[k] += r["shadow"][r["seats"].index(tag)][k]
    by_deck: dict[str, list[int]] = {}  # round 5: the opponent's deck -> [wins, games]
    turns = []
    for res in st["results"].values():
        if res.get("decks"):
            d = res["decks"][1 - res["seats"].index(tag)] or "CyberDragon"
            k = by_deck.setdefault(d, [0, 0])
            k[0] += int(res.get("winner") == res["seats"].index(tag))
            k[1] += 1
        if "error" not in res:
            turns.append(res.get("turns", 0))
    return {"name": st["name"], "tag": tag, "games": n, "loop_breaks": loop_breaks, "wins": wins, "losses": losses, "draws": draws, "errors": errors,
            "win_rate": round(wins / max(n, 1), 3), "replays": replays, "shadow": shadow,
            "by_deck": {d: round(w / g, 3) for d, (w, g) in sorted(by_deck.items())},
            "mean_turns": round(sum(turns) / max(1, len(turns)), 1),
            "errors_detail": [r["error"] for r in st["results"].values() if "error" in r][:5]}


# ------------------------------------------------------------------ train

def train() -> None:
    R = recipe()
    sc = Scorer(MODEL, chat=False, lora_r=R["lora_r"], budget=int(os.environ.get("YGO_BUDGET", "24000")))
    model = sc.model
    if DEVICE != "cuda":  # the 3060 has room for MICRO x ~500-token activations; checkpointing would cost ~30%
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = list(lora_params(model).values())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=R["lr"], weight_decay=0.0)
    s = {"step": 0, "gpu_s": 0.0, "evals": [], "pool": [], "last_ckpt_gpu_s": 0.0, "games": 0, "errors": 0,
         "ret_mean": 0.0, "ret_var": 1.0, "jobs": [], "seed": 1, "recipe": R, "decisions": 0}
    buffer: list[dict] = []   # finished decisions waiting for an update
    traj: dict[str, list] = {}  # "job:idx:seat" -> decisions of a game still running
    if (CKPT / "state.json").exists():
        s.update(json.loads((CKPT / "state.json").read_text()))
        blob = torch.load(CKPT / "ckpt.pt", map_location=DEVICE, weights_only=False)
        with torch.no_grad():
            for n, p in lora_params(model).items():
                p.copy_(blob["lora"][n])
        opt.load_state_dict(blob["opt"])
        buffer, traj = blob["buffer"], blob["traj"]
        log("resumed", step=s["step"], msg=f"resumed step={s['step']}", gpu_h=round(s["gpu_s"] / 3600, 3),
            buffer=len(buffer), live=len(traj))
    else:
        if WARM:
            blob = torch.load(WARM, map_location=DEVICE)
            with torch.no_grad():
                for n, p in lora_params(model).items():
                    p.copy_(blob[n])
            s["warm"] = WARM
        log("fresh", step=0, model=MODEL, recipe=R, warm=WARM)
    pool = {p["name"]: torch.load(CKPT / "pool" / f"{p['name']}.pt", map_location=DEVICE) for p in s["pool"]}

    def save_ckpt(reason: str) -> None:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"lora": {n: p.detach() for n, p in lora_params(model).items()}, "opt": opt.state_dict(),
                    "buffer": buffer, "traj": traj}, CKPT / "ckpt.tmp")
        os.replace(CKPT / "ckpt.tmp", CKPT / "ckpt.pt")
        s["last_save_gpu_s"] = s["gpu_s"]
        save_json(CKPT / "state.json", s)
        log("checkpoint", step=s["step"], msg=f"checkpoint step={s['step']}", reason=reason)

    def add_pool() -> None:
        name = f"step-{s['step']}"
        (CKPT / "pool").mkdir(parents=True, exist_ok=True)
        sd = {n: p.detach().clone() for n, p in lora_params(model).items()}
        torch.save(sd, CKPT / "pool" / f"{name}.pt")
        s["pool"].append({"name": name, "step": s["step"], "gpu_h": round(s["gpu_s"] / 3600, 3)})
        pool[name] = sd
        while len(pool) > POOL:
            old = next(p["name"] for p in s["pool"] if p["name"] in pool)
            del pool[old]  # file stays on disk for the Phase 3 table
        s["pool_live"] = list(pool)
        s["last_ckpt_gpu_s"] = s["gpu_s"]

    clock = [time.time()]

    def tick() -> None:
        now = time.time()
        s["gpu_s"] += now - clock[0]
        clock[0] = now

    def shaping_coef() -> float:
        return max(0.0, R["shaping"] * (1 - s["gpu_s"] / 3600 / (CAP_H / 2)))

    def do_eval() -> float | None:
        """200 greedy games vs each of EVAL_OPPS (WindBot, then random; each resumable). Returns the WindBot win rate."""
        out = {}
        for opp in EVAL_OPPS:
            part = s.setdefault("eval_parts", {})
            key = f"{s['step']}:{opp}"
            if key in part:
                out[opp] = part[key]
                continue
            try:
                res = run_match(f"eval-step{s['step']}-{opp}", seats_vs("eval", opp, EVAL_GAMES),
                                {"eval": lambda rs: choose(sc, rs, greedy=True)}, DATA / "eval-state.json",
                                seed=1000 + s["step"], on_tick=tick)
            except Stopped:
                return None
            traj.clear()  # games that ran during the eval got stand-in answers; their trajectories are incomplete
            if res is None:
                return None
            (DATA / "eval-state.json").unlink(missing_ok=True)
            part[key] = out[opp] = res
            save_json(CKPT / "state.json", s)
        w, r = out["windbot"], out.get("random")
        parts = list(out.values())
        gpu_h = round(s["gpu_s"] / 3600, 2)
        s["evals"].append({"step": s["step"], "gpu_h": gpu_h, "vs_windbot": w["win_rate"],
                           "vs_random": r["win_rate"] if r else None, "games": w["games"],
                           "errors": sum(x["errors"] for x in parts), "replays": sum((x["replays"] for x in parts), [])})
        s["eval_parts"] = {}
        vs_r = f" vs_random={r['win_rate']:.3f}" if r else ""
        log("eval", step=s["step"], gpu_h=gpu_h, vs_windbot=w["win_rate"], vs_random=r["win_rate"] if r else None,
            games=w["games"], games_random=r["games"] if r else 0, errors=sum(x["errors"] for x in parts),
            wins_windbot=w["wins"], draws_windbot=w["draws"], loop_breaks=sum(x["loop_breaks"] for x in parts),
            errors_detail=sum((x["errors_detail"] for x in parts), [])[:3],
            msg=f"eval step={s['step']} gpu_h={gpu_h} vs_windbot={w['win_rate']:.3f}{vs_r} games={w['games']}")
        save_json(CKPT / "state.json", s)
        return w["win_rate"]

    def submit_games(n: int) -> None:
        games = []
        for _ in range(n):
            u = random.random()
            if u < R["random_share"]:
                opp = "random"
            elif u < R["random_share"] + R["windbot_share"]:
                opp = "windbot"
            elif u < R["random_share"] + R["windbot_share"] + R["pool_share"] and pool:
                opp = "pool:" + random.choice(list(pool))
            else:
                opp = "learner"
            seats = ["learner", opp] if random.random() < 0.5 else [opp, "learner"]
            games.append({"seats": seats, "replay": random.random() < 0.002, "live": not games})
        s["seed"] += 1
        job = server_call("/job", {"name": f"train-step{s['step']}", "seed": s["seed"], "games": games})["job"]
        s["jobs"] = (s["jobs"] + [job])[-200:]

    def finish_game(res: dict) -> None:
        s["games"] += 1
        keys = [f"{res['job']}:{res['idx']}:{k}" for k in (0, 1)]
        if "error" in res:
            s["errors"] += 1
            log("game_error", job=res["job"], idx=res["idx"], error=res["error"][:300])
            for k in keys:
                traj.pop(k, None)
            return
        c = shaping_coef()
        for seat, k in enumerate(keys):
            ds = traj.pop(k, None)
            if not ds:
                continue
            won = res["winner"] == seat
            R_ = 1.0 if won else -1.0  # a draw counts as a loss
            phi_end = (res["lp"][seat] - res["lp"][1 - seat]) / 8000
            looped = {t for st_, t in res.get("loop_turns", []) if st_ == seat}
            for d in ds:
                if random.random() < KEEP_P:
                    d["ret"] = R_ + c * (phi_end - d["phi"]) - (LOOP_PENALTY if d.get("turn") in looped else 0.0)
                    buffer.append(d)
        del buffer[: max(0, len(buffer) - BUFFER_CAP)]

    def update() -> dict | None:
        try:
            return update_once()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            model.eval()
            torch.cuda.empty_cache()
            s["micro"] = max(1, s.get("micro", MICRO) // 2)  # long card-text prompts: smaller micro-batches from now on
            log("oom_skip", step=s["step"], micro=s["micro"])  # this batch is dropped; the next one tries again
            return None

    def update_once() -> dict:
        random.shuffle(buffer)
        batch = [buffer.pop() for _ in range(BATCH)]
        rets = torch.tensor([d["ret"] for d in batch])
        s["ret_mean"] = 0.99 * s["ret_mean"] + 0.01 * float(rets.mean())
        s["ret_var"] = 0.99 * s["ret_var"] + 0.01 * float(((rets - s["ret_mean"]) ** 2).mean())
        adv = (rets - s["ret_mean"]) / math.sqrt(s["ret_var"] + 1e-6)
        model.train()
        opt.zero_grad(set_to_none=True)
        tot = {"pg": 0.0, "ent": 0.0, "clipfrac": 0.0, "kl": 0.0}
        micro = s.get("micro", MICRO)
        for i in range(0, BATCH, micro):
            mb = batch[i:i + micro]
            lg = sc.logits([d["prompt"] for d in mb], [d["n"] for d in mb], grad=True)
            loss = 0.0
            for d, l, a in zip(mb, lg, adv[i:i + micro]):
                lp = torch.log_softmax(l / R["temp"], -1)
                ratio = torch.exp(lp[d["a"]] - d["logp"])
                a = a.to(DEVICE)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - R["clip"], 1 + R["clip"]) * a)
                ent = -(lp.exp() * lp).sum()
                loss = loss + (pg - R["entropy"] * ent) / BATCH
                tot["pg"] += pg.item() / BATCH
                tot["ent"] += ent.item() / BATCH
                tot["clipfrac"] += float(abs(ratio.item() - 1) > R["clip"]) / BATCH
                tot["kl"] += (d["logp"] - lp[d["a"]].item()) / BATCH
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.eval()
        for d in batch:  # PPO epochs: back into the buffer for another pass (old logp kept, ratio clipped)
            d["epochs"] = d.get("epochs", 0) + 1
            if d["epochs"] < PPO_EPOCHS:
                buffer.insert(0, d)
        return {k: round(v, 4) for k, v in tot.items()}

    try:
        if LABELS_FILE and not s.get("sl_done"):
            imitate(s, sc, model, opt, params, save_ckpt, add_pool, tick)
        if s.get("sl_done") or not LABELS_FILE:
            train_loop(s, sc, model, opt, buffer, traj, pool, save_ckpt, add_pool, tick, shaping_coef, do_eval,
                       submit_games, finish_game, update)
    except Stopped:
        pass
    if not (DATA / "done-train").exists():
        save_ckpt("stop")


def train_loop(s, sc, model, opt, buffer, traj, pool, save_ckpt, add_pool, tick, shaping_coef, do_eval,
               submit_games, finish_game, update) -> None:
    R = s["recipe"]
    server_call("/cancel", {})  # games queued by a previous run that never started
    if not s["evals"] and EVAL_GAMES > 0:
        if do_eval() is None:
            return
    last_eval_gpu_s = s["evals"][-1]["gpu_h"] * 3600 if s["evals"] else 0.0
    t_log = time.time()
    while not STOP:
        tick()
        st = server_call("/status")
        if st["queued"] + st["running"] < LIVE_GAMES and len(buffer) < BUFFER_CAP // 2:
            submit_games(LIVE_GAMES - st["queued"] - st["running"])
        nxt = server_call("/next?max=256&wait=2&gather=50", timeout=40)
        reqs = nxt["requests"]
        answers = []
        mine = [r for r in reqs if r["job"] in s["jobs"]]
        answers += [{"id": r["id"], "choice": random.randrange(r["n"])} for r in reqs if r["job"] not in s["jobs"]]
        by_tag: dict[str, list] = {}
        for r in mine:
            by_tag.setdefault(r["policy"], []).append(r)
        for tag, rs in by_tag.items():
            state = pool.get(tag[5:]) if tag.startswith("pool:") else None
            if tag.startswith("pool:") and state is None:
                state = next(iter(pool.values()), None)  # pool entry rotated out: use the oldest live one
            picks = choose(sc, rs, greedy=False, temp=R["temp"], state=state)
            for r, (a, lp) in zip(rs, picks):
                answers.append({"id": r["id"], "choice": a})
                if tag == "learner":
                    traj.setdefault(f"{r['job']}:{r['idx']}:{r['seat']}", []).append(
                        {"prompt": r["prompt"], "n": r["n"], "a": a, "logp": lp, "phi": (r["lp"][0] - r["lp"][1]) / 8000,
                         "turn": r.get("turn")})
                    s["decisions"] += 1
        if answers:
            server_call("/answer", answers)
        for res in nxt["results"]:
            if res["job"] in s["jobs"]:
                finish_game(res)
        for _ in range(4):  # drain a full buffer (e.g. the one saved across a pause) a few updates at a time
            if len(buffer) < BATCH or STOP:
                break
            stats = update()
            if stats is None:
                continue
            s["step"] += 1
            log("step", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), buffer=len(buffer), live=len(traj),
                games=s["games"], shaping=round(shaping_coef(), 3), **stats)
        tick()
        if s["gpu_s"] - s["last_ckpt_gpu_s"] >= CKPT_EVERY_S:
            add_pool()
            save_ckpt("periodic")
        elif s["gpu_s"] - s.get("last_save_gpu_s", 0.0) >= SAVE_EVERY_S:
            save_ckpt("periodic")
        if time.time() - t_log > 300:
            t_log = time.time()
            active = set(int(j) for j in server_call("/status").get("active", {}))
            for k in [k for k in traj if int(k.split(":")[0]) not in active]:
                del traj[k]  # its game was lost (server restart) or ran during an eval
            log("progress", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), games=s["games"], errors=s["errors"],
                decisions=s["decisions"], buffer=len(buffer))
        done = s["gpu_s"] >= s.get("cap_h", CAP_H) * 3600
        if (s["gpu_s"] - last_eval_gpu_s >= EVAL_EVERY_H * 3600 or done) and EVAL_GAMES > 0 and not STOP:
            server_call("/cancel", {})
            wr = do_eval()
            if wr is None:
                break
            last_eval_gpu_s = s["gpu_s"]
            add_pool()
            save_ckpt("eval")
            ev = s["evals"]
            if done and wr < TARGET and CAP2_H > s.get("cap_h", CAP_H) and len(ev) >= 2 and ev[-1]["vs_windbot"] > ev[-2]["vs_windbot"]:
                s["cap_h"] = CAP2_H  # still climbing at the cap: one extension, then stop whatever it reached
                log("cap_extend", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 2), cap_h=CAP2_H,
                    last=ev[-1]["vs_windbot"], before=ev[-2]["vs_windbot"],
                    msg=f"cap_extend to {CAP2_H} GPU-hours: {ev[-1]['vs_windbot']:.3f} > {ev[-2]['vs_windbot']:.3f}")
                save_json(CKPT / "state.json", s)
                continue
            if wr >= TARGET or done:
                reason = "target" if wr >= TARGET else "cap"
                server_call("/cancel", {})
                log("train_stop", reason=reason, step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 2), vs_windbot=wr,
                    msg=f"train_stop reason={reason}")
                (DATA / "done-train").write_text(reason)
                return


# ------------------------------------------------------------------ round 3: imitation and the card-text check

def imitate(s, sc, model, opt, params, save_ckpt, add_pool, tick) -> None:
    """Supervised cross-entropy on the labels' picks (held-out games excluded), SL_EPOCHS passes in a fixed
    shuffled order. Resumable: the position is in the state. Ends with a pool snapshot and held-out accuracy."""
    train_set = load_labels(held=False)
    order = list(range(len(train_set)))
    random.Random(3).shuffle(order)
    total = int(len(order) * SL_EPOCHS)
    for g in opt.param_groups:
        g["lr"] = SL_LR
    pos = s.get("sl_pos", 0)
    log("sl_start", labels=len(train_set), steps=total // BATCH, pos=pos, lr=SL_LR, card_text=CARD_TEXT)
    t_save = time.time()
    while pos + BATCH <= total:
        if STOP:
            raise Stopped()
        batch = [train_set[order[k % len(order)]] for k in range(pos, pos + BATCH)]
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_t = acc = 0.0
        micro = s.get("micro", MICRO)
        try:
            for i in range(0, BATCH, micro):
                mb = batch[i:i + micro]
                lg = sc.logits([d["prompt"] for d in mb], [d["n"] for d in mb], grad=True)
                loss = sum(-torch.log_softmax(l, -1)[d["choice"]] for d, l in zip(mb, lg)) / BATCH
                loss.backward()
                loss_t += loss.item()
                acc += sum(int(torch.argmax(l)) == d["choice"] for d, l in zip(mb, lg)) / BATCH
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            model.eval()
            torch.cuda.empty_cache()
            s["micro"] = max(1, micro // 2)
            log("oom_skip", step=s["step"], micro=s["micro"])
            continue  # same batch again with smaller micro-batches
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.eval()
        pos += BATCH
        s["sl_pos"] = pos
        s["step"] += 1
        tick()
        log("step", phase="sl", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), loss=round(loss_t, 4),
            acc=round(acc, 3), pos=pos, of=total)
        if time.time() - t_save > SAVE_EVERY_S:
            save_ckpt("periodic")
            t_save = time.time()
        if SL_CAP_H and s["gpu_s"] >= SL_CAP_H * 3600:
            log("sl_cap", step=s["step"], pos=pos, of=total, gpu_h=round(s["gpu_s"] / 3600, 3),
                msg=f"sl_cap at {SL_CAP_H} GPU-h: {pos} of {total} labels")
            break
    held = load_labels(held=True, src="windbot")
    random.Random(0).shuffle(held)
    held = held[:2500]
    hits = sum(int(torch.argmax(l)) == d["choice"] for d, l in zip(held, sc.logits([d["prompt"] for d in held], [d["n"] for d in held])))
    for g in opt.param_groups:
        g["lr"] = s["recipe"]["lr"]
    opt.state.clear()  # PPO starts with fresh Adam moments at its own learning rate
    s["sl_done"] = True
    s["sl_heldout_acc"] = round(hits / max(1, len(held)), 4)
    tick()
    add_pool()
    save_ckpt("sl_done")
    log("sl_done", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), heldout_acc=s["sl_heldout_acc"], heldout=len(held),
        msg=f"sl_done step={s['step']} heldout_acc={s['sl_heldout_acc']:.3f} on {len(held)} held-out WindBot decisions")


def textcheck() -> None:
    """No training: YGO_WARM's top-1 agreement with WindBot on held-out decisions, with and without card text."""
    n = int(os.environ.get("YGO_CHECK_N", "2500"))
    held = load_labels(held=True, src="windbot")
    random.Random(0).shuffle(held)
    held = held[:n]
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    if WARM:
        blob = torch.load(WARM, map_location=DEVICE)
        with torch.no_grad():
            for k, p in lora_params(sc.model).items():
                p.copy_(blob[k])
    acc = {}
    for mode in ("full", "none"):
        hits = 0
        for i in range(0, len(held), 256):
            if STOP:
                return
            part = held[i:i + 256]
            lg = sc.logits([render_cards(d["prompt"], mode) for d in part], [d["n"] for d in part])
            hits += sum(int(torch.argmax(l)) == d["choice"] for d, l in zip(part, lg))
        acc[mode] = hits / len(held)
        log("textcheck_part", mode=mode, decisions=len(held), acc=round(acc[mode], 4))
    # chance level for reference: the mean of 1/n
    chance = sum(1 / d["n"] for d in held) / len(held)
    pick = "full" if acc["full"] >= acc["none"] + 0.02 else "oneliner"
    save_json(DATA / "textcheck.json", {"warm": WARM, "decisions": len(held), "acc_text": acc["full"],
                                        "acc_notext": acc["none"], "chance": chance, "card_text": pick})
    log("textcheck", decisions=len(held), chance=round(chance, 4),
        msg=f"check decisions={len(held)} acc_text={acc['full']:.4f} acc_notext={acc['none']:.4f} card_text={pick}")
    (DATA / "done-textcheck").write_text(pick)


# ------------------------------------------------------------------ round 4: pre-registered matches

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for k wins in n games."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def paired_vs(tag: str, opp: str, games: int, seed: int) -> list[dict]:
    """Round 4: games in pairs with one seed, the seats swapped, so both players get the same deals."""
    out = []
    for g in range(games):
        gs = (seed * 1_000_003 + g // 2) & (2**63 - 1)
        out.append({"seats": [tag, opp] if g % 2 == 0 else [opp, tag], "seed": gs})
    return out


def held_game(key) -> bool:
    import hashlib
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 10 == 0


def lp_phi(prompt: str) -> float:
    m = re.search(r"Life points: me (\d+), opponent (\d+)", prompt)
    return (int(m.group(1)) - int(m.group(2))) / 8000 if m else 0.0


def match() -> None:
    """Round 4: checkpoint YGO_MATCH_A (a pool .pt) plays YGO_MATCH_B ("windbot" or a .pt) for YGO_MATCH_GAMES greedy
    games, half as first player, draws count as losses. Writes <YGO_DATA>/result.json with the Wilson 95% interval
    and the GPU seconds spent (kept across pauses in the match state).
    YGO_MATCH_B_FMT=r3: B sees round-3 prompts (no history, round-3 one-liners). YGO_MATCH_A_FMT=r3 likewise for A.
    YGO_MATCH_PAIRED=1: paired seeds. YGO_MATCH_SHADOW=1: A's seat has a shadow WindBot (DAgger labels on the game host).
    YGO_MATCH_VALUE_LOG=file: up to 40 of A's decisions per game with the game's outcome (critic data)."""
    name = os.environ["YGO_MATCH_NAME"]
    a_path, b = os.environ["YGO_MATCH_A"], os.environ["YGO_MATCH_B"]
    games = int(os.environ.get("YGO_MATCH_GAMES", "500"))
    seed = int(os.environ.get("YGO_MATCH_SEED", "4000"))
    fmt = lambda k: R3_FMT if os.environ.get(k) == "r3" else None
    t0 = time.time()
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    load = lambda p: torch.load(p, map_location=DEVICE)
    states = {"ck:a": load(a_path)}
    vlog_path = os.environ.get("YGO_MATCH_VALUE_LOG", "")
    pending: dict[str, list] = {}  # "job:idx" -> A's decisions (value log)

    def play_a(rs):
        out = choose(sc, rs, greedy=True, state=states["ck:a"], fmt=fmt("YGO_MATCH_A_FMT"))
        if vlog_path:
            for r in rs:
                pending.setdefault(f"{r['job']}:{r['idx']}", []).append({"prompt": r["prompt"], "n": r["n"], "seat": r["seat"]})
        return out
    players = {"ck:a": play_a}
    me = "shadow:ck:a" if os.environ.get("YGO_MATCH_SHADOW") == "1" else "ck:a"
    opp, opp_decks = b, None
    if b == "pool":  # round 5: WindBot piloting the 5 pool decks in turn
        opp, opp_decks = "windbot", POOL_DECKS
    elif b != "windbot":
        opp = "ck:b"
        b_model = os.environ.get("YGO_MATCH_B_MODEL", MODEL)  # round 5: B may be a 0.5B model when A is the 1.5B
        sc_b = sc if b_model == MODEL else Scorer(b_model, chat=False, lora_r=16)
        states[opp] = load(b)
        players[opp] = lambda rs: choose(sc_b, rs, greedy=True, state=states["ck:b"], fmt=fmt("YGO_MATCH_B_FMT"))
    gpu_f = DATA / "gpu_s.json"
    prev = json.loads(gpu_f.read_text())["gpu_s"] if gpu_f.exists() else 0.0
    clock = [t0]
    st_file = DATA / "match-state.json"

    def tick() -> None:
        now = time.time()
        if now - clock[0] <= 30:
            return
        save_json(gpu_f, {"gpu_s": prev + now - t0})
        clock[0] = now
        if vlog_path and pending and st_file.exists():
            for res in json.loads(st_file.read_text())["results"].values():
                key = f"{res['job']}:{res['idx']}"
                ds = pending.pop(key, None)
                if not ds or "error" in res:
                    continue
                step = max(1, len(ds) // 40)
                with open(vlog_path, "a") as f:
                    for d in ds[::step][:40]:
                        f.write(json.dumps({"prompt": d["prompt"], "n": d["n"], "seed": key, "player": d["seat"],
                                            "won": int(res["winner"] == d["seat"]), "held": held_game(key)}) + "\n")

    gl = paired_vs(me, opp, games, seed) if os.environ.get("YGO_MATCH_PAIRED") == "1" else seats_vs(me, opp, games)
    gl = with_decks(gl, me, opp_decks)
    res = run_match(name, gl, players, st_file, seed=seed, on_tick=tick)
    clock[0] = 0
    tick()
    gpu_s = prev + time.time() - t0
    save_json(gpu_f, {"gpu_s": gpu_s})
    if res is None:
        return
    lo, hi = wilson(res["wins"], res["games"])
    out = {"a": a_path, "b": b, **res, "ci95": [lo, hi], "gpu_s": round(gpu_s, 1)}
    if me.startswith("shadow:"):
        asked, hit, dis = res["shadow"]
        out["coverage"] = round(hit / max(1, asked), 4)
        out["disagree"] = round(dis / max(1, hit), 4)
    save_json(DATA / "result.json", out)
    log("match_done", msg=f"match name={name} win_rate={res['win_rate']:.3f} ci95=[{lo:.3f},{hi:.3f}] "
                          f"games={res['games']} errors={res['errors']} gpu_h={gpu_s / 3600:.2f}")
    (DATA / "done-match").write_text("ok")


def gpu_clock():
    """(tick, total): GPU seconds of a mode kept in <YGO_DATA>/gpu_s.json across pauses."""
    f = DATA / "gpu_s.json"
    base = json.loads(f.read_text())["gpu_s"] if f.exists() else 0.0
    t0 = time.time()
    last = [t0]

    def tick(force: bool = False) -> float:
        now = time.time()
        if force or now - last[0] > 30:
            save_json(f, {"gpu_s": base + now - t0})
            last[0] = now
        return base + now - t0
    return tick


def load_lora(sc: Scorer, path: str) -> None:
    blob = torch.load(path, map_location=DEVICE)
    with torch.no_grad():
        for n, p in lora_params(sc.model).items():
            p.copy_(blob[n])


def sl() -> None:
    """Round 4 re-imitation (and DAgger's aggregate SL): YGO_WARM plus supervised cross-entropy on YGO_LABELS_FILE
    (held-out games excluded), SL_EPOCHS at SL_LR. Resumable. Writes <YGO_DATA>/final.pt and result.json."""
    tick = gpu_clock()
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    model = sc.model
    params = list(lora_params(model).values())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=SL_LR, weight_decay=0.0)
    s = {"step": 0, "gpu_s": 0.0, "recipe": {"lr": SL_LR}, "pool": []}
    if (CKPT / "state.json").exists():
        s.update(json.loads((CKPT / "state.json").read_text()))
        blob = torch.load(CKPT / "ckpt.pt", map_location=DEVICE, weights_only=False)
        with torch.no_grad():
            for n, p in lora_params(model).items():
                p.copy_(blob["lora"][n])
        opt.load_state_dict(blob["opt"])
        log("resumed", step=s["step"], msg=f"resumed step={s['step']}")
    else:
        if WARM:  # round 5: no WARM = a fresh LoRA (the 1.5B trial imitates from scratch)
            load_lora(sc, WARM)
        log("fresh", warm=WARM, labels=LABELS_FILE, lr=SL_LR, model=MODEL)

    def save_ckpt(reason: str) -> None:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"lora": {n: p.detach() for n, p in lora_params(model).items()}, "opt": opt.state_dict()}, CKPT / "ckpt.tmp")
        os.replace(CKPT / "ckpt.tmp", CKPT / "ckpt.pt")
        save_json(CKPT / "state.json", s)
        log("checkpoint", step=s["step"], msg=f"checkpoint step={s['step']}", reason=reason)

    def add_pool() -> None:
        torch.save({n: p.detach().clone() for n, p in lora_params(model).items()}, DATA / "final.pt")

    try:
        imitate(s, sc, model, opt, params, save_ckpt, add_pool, lambda: s.__setitem__("gpu_s", tick()))
    except Stopped:
        save_ckpt("stop")
        tick(True)
        return
    gpu_s = tick(True)
    save_json(DATA / "result.json", {"warm": WARM, "labels": LABELS_FILE, "steps": s["step"], "lr": SL_LR,
                                     "heldout_acc": s["sl_heldout_acc"], "gpu_s": round(gpu_s, 1)})
    (DATA / "done-sl").write_text("ok")


class ValueHead(torch.nn.Module):
    """Round 4 critic: P(win) from the policy's final hidden state at the answer position (features detached, so
    the value loss never moves the policy)."""

    def __init__(self, width: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f.float()).squeeze(-1)


def auc(p: list[float], y: list[int]) -> float:
    pairs = sorted(zip(p, y))
    pos = sum(y)
    neg = len(y) - pos
    rank_sum, i = 0.0, 0
    while i < len(pairs):  # average ranks over ties
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(yy for _, yy in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / max(1, pos * neg)


def values() -> None:
    """Round 4 critic and its gate. YGO_WARM's features for every decision in YGO_VALUE_FILES (comma list: WindBot
    self-play and model-vs-WindBot decisions with their game's outcome), a value head trained on the games not held
    out, scored on the held-out games against an LP-only predictor (logistic in the LP difference, fit on the same
    training games). critic=on iff auc >= 0.75 and brier <= 0.9 x brier_lp. Writes vhead.pt, result.json."""
    tick = gpu_clock()
    rows = []
    for f in os.environ["YGO_VALUE_FILES"].split(","):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh]
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    load_lora(sc, WARM)
    sc.feature_hook()
    feat_f = DATA / "features.pt"
    if feat_f.exists():
        X = torch.load(feat_f, map_location=DEVICE)
    else:
        X = []
        for i in range(0, len(rows), 256):
            if STOP:
                tick(True)
                return
            part = rows[i:i + 256]
            fs: list = []
            with torch.no_grad():
                sc.logits([r["prompt"] for r in part], [r["n"] for r in part], feats=fs)
            X += fs
            tick()
        X = torch.stack(X)
        torch.save(X, feat_f)
    y = torch.tensor([float(r["won"]) for r in rows], device=DEVICE)
    phi = torch.tensor([lp_phi(r["prompt"]) for r in rows], device=DEVICE)
    held = torch.tensor([bool(r["held"]) for r in rows], device=DEVICE)
    tr, te = ~held, held
    torch.manual_seed(4)
    head = ValueHead(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = X[tr], y[tr]
    for epoch in range(int(os.environ.get("YGO_VALUE_EPOCHS", "8"))):
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for k in range(0, len(perm), 256):
            idx = perm[k:k + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    w = torch.zeros(2, device=DEVICE, requires_grad=True)  # the LP-only predictor
    lopt = torch.optim.Adam([w], lr=0.05)
    for _ in range(2000):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(w[0] * phi[tr] + w[1], y[tr])
        lopt.zero_grad()
        loss.backward()
        lopt.step()
    with torch.no_grad():
        p = torch.sigmoid(head(X[te]))
        p_lp = torch.sigmoid(w[0] * phi[te] + w[1])
        brier = float(((p - y[te]) ** 2).mean())
        brier_lp = float(((p_lp - y[te]) ** 2).mean())
        a = auc(p.tolist(), [int(v) for v in y[te].tolist()])
        a_lp = auc(p_lp.tolist(), [int(v) for v in y[te].tolist()])
    games = len({r["seed"] for r in rows})
    held_games = len({r["seed"] for r in rows if r["held"]})
    on = a >= 0.75 and brier <= 0.9 * brier_lp
    torch.save(head.state_dict(), DATA / "vhead.pt")
    gpu_s = tick(True)
    out = {"warm": WARM, "auc": round(a, 4), "brier": round(brier, 4), "brier_lp": round(brier_lp, 4), "auc_lp": round(a_lp, 4),
           "games": games, "held_games": held_games, "decisions": len(rows), "held_decisions": int(te.sum()),
           "critic": "on" if on else "off", "gpu_s": round(gpu_s, 1)}
    save_json(DATA / "result.json", out)
    log("values", msg=f"value auc={a:.4f} brier={brier:.4f} brier_lp={brier_lp:.4f} games={games} "
                      f"(held-out {held_games} games, {int(te.sum())} decisions; auc_lp={a_lp:.4f})")
    (DATA / "done-values").write_text(out["critic"])


def values2() -> None:
    """Round 5 bigger critic: a value head on its own trainable LoRA trunk (initialised from YGO_WARM), trained
    with BCE on game outcomes: up to YGO_VALUE_PER_GAME decisions of every training game in YGO_VALUE_FILES, one
    pass (YGO_VALUE_EPOCHS), resumable. Scored on the held-out games against the LP-only predictor; the gate is
    round 4's: critic=on iff auc >= 0.75 and brier <= 0.9 x brier_lp. Writes vtrunk.pt + vhead.pt, result.json."""
    tick = gpu_clock()
    per_game = int(os.environ.get("YGO_VALUE_PER_GAME", "16"))
    by_game: dict = {}
    for f in os.environ["YGO_VALUE_FILES"].split(","):
        with open(f) as fh:
            for l in fh:
                r = json.loads(l)
                by_game.setdefault(f"{f}|{r['seed']}", []).append(r)
    rows = []
    for k in sorted(by_game):
        ds = by_game[k]
        step = max(1, len(ds) // per_game)
        rows += [dict(r, game=k) for r in ds[::step][:per_game]]
    train_rows = [r for r in rows if not r["held"]]
    held_rows = [r for r in rows if r["held"]]
    random.Random(5).shuffle(train_rows)
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    load_lora(sc, WARM)
    sc.feature_hook()
    model = sc.model
    head = ValueHead(model.config.hidden_size).to(DEVICE)
    params = list(lora_params(model).values())
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": params, "lr": float(os.environ.get("YGO_VALUE_LR", "5e-5"))},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.0)
    st = {"pos": 0}
    if (CKPT / "state.json").exists():
        st = json.loads((CKPT / "state.json").read_text())
        blob = torch.load(CKPT / "ckpt.pt", map_location=DEVICE, weights_only=False)
        with torch.no_grad():
            for n, p_ in lora_params(model).items():
                p_.copy_(blob["lora"][n])
        head.load_state_dict(blob["head"])
        opt.load_state_dict(blob["opt"])
        log("resumed", pos=st["pos"], msg=f"resumed pos={st['pos']}")
    total = int(len(train_rows) * float(os.environ.get("YGO_VALUE_EPOCHS", "1")))

    def save(reason: str) -> None:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"lora": {n: p_.detach() for n, p_ in lora_params(model).items()}, "head": head.state_dict(),
                    "opt": opt.state_dict()}, CKPT / "ckpt.tmp")
        os.replace(CKPT / "ckpt.tmp", CKPT / "ckpt.pt")
        save_json(CKPT / "state.json", st)
        log("checkpoint", msg=f"checkpoint pos={st['pos']}", reason=reason)

    log("values2_start", rows=len(rows), train=len(train_rows), held=len(held_rows), games=len(by_game), total=total)
    t_save = time.time()
    micro = MICRO
    while st["pos"] + BATCH <= total:
        if STOP:
            save("stop")
            tick(True)
            return
        batch = [train_rows[k % len(train_rows)] for k in range(st["pos"], st["pos"] + BATCH)]
        model.train()
        opt.zero_grad(set_to_none=True)
        lt = 0.0
        try:
            for i in range(0, BATCH, micro):
                mb = batch[i:i + micro]
                fs: list = []
                sc.logits([r["prompt"] for r in mb], [r["n"] for r in mb], grad=True, feats=fs)
                y = torch.tensor([float(r["won"]) for r in mb], device=DEVICE)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(torch.stack(fs)), y) * len(mb) / BATCH
                loss.backward()
                lt += loss.item()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            model.eval()
            torch.cuda.empty_cache()
            micro = max(1, micro // 2)
            log("oom_skip", micro=micro)
            continue
        torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
        opt.step()
        model.eval()
        st["pos"] += BATCH
        tick()
        if st["pos"] % (BATCH * 50) == 0:
            log("step", phase="values2", pos=st["pos"], of=total, loss=round(lt, 4))
        if time.time() - t_save > SAVE_EVERY_S:
            save("periodic")
            t_save = time.time()
    preds = []
    for i in range(0, len(held_rows), 64):
        if STOP:
            save("stop")
            tick(True)
            return
        part = held_rows[i:i + 64]
        fs = []
        with torch.no_grad():
            sc.logits([r["prompt"] for r in part], [r["n"] for r in part], feats=fs)
            preds += torch.sigmoid(head(torch.stack(fs))).tolist()
        tick()
    y_tr = torch.tensor([float(r["won"]) for r in train_rows], device=DEVICE)
    phi_tr = torch.tensor([lp_phi(r["prompt"]) for r in train_rows], device=DEVICE)
    w = torch.zeros(2, device=DEVICE, requires_grad=True)  # the LP-only predictor, fit on the same training games
    lopt = torch.optim.Adam([w], lr=0.05)
    for _ in range(2000):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(w[0] * phi_tr + w[1], y_tr)
        lopt.zero_grad()
        loss.backward()
        lopt.step()
    y = [int(r["won"]) for r in held_rows]
    with torch.no_grad():
        p_lp = torch.sigmoid(w[0] * torch.tensor([lp_phi(r["prompt"]) for r in held_rows], device=DEVICE) + w[1]).tolist()
    brier = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)
    brier_lp = sum((p - t) ** 2 for p, t in zip(p_lp, y)) / len(y)
    a, a_lp = auc(preds, y), auc(p_lp, y)
    on = a >= 0.75 and brier <= 0.9 * brier_lp
    torch.save({n: p_.detach().clone() for n, p_ in lora_params(model).items()}, DATA / "vtrunk.pt")
    torch.save(head.state_dict(), DATA / "vhead.pt")
    gpu_s = tick(True)
    games = len(by_game)
    held_games = len({r["game"] for r in held_rows})
    out = {"warm": WARM, "auc": round(a, 4), "brier": round(brier, 4), "brier_lp": round(brier_lp, 4), "auc_lp": round(a_lp, 4),
           "games": games, "held_games": held_games, "decisions": len(rows), "held_decisions": len(held_rows),
           "critic": "on" if on else "off", "gpu_s": round(gpu_s, 1)}
    save_json(DATA / "result.json", out)
    log("values", msg=f"value auc={a:.4f} brier={brier:.4f} brier_lp={brier_lp:.4f} games={games} "
                      f"(held-out {held_games} games, {len(held_rows)} decisions; auc_lp={a_lp:.4f})")
    (DATA / "done-values2").write_text(out["critic"])


CA_K = float(os.environ.get("YGO_CA_K", "0.02"))


def ca_phi(counts) -> float:
    """Round 5 card-advantage potential (the round-5 research formula) from the runner's public counts
    [my hand, monsters, spells/traps, GY, banished, opp hand, opp face-up monsters, opp face-down cards, opp GY,
    opp banished]: A_me = H + M + 0.75 ST + 0.1 min(GY, 10), A_opp = H + M_up + 0.85 FD + 0.1 min(GY, 10),
    phi = CA_K x clip(A_me - A_opp, -8, 8). Ng's shaping uses phi(s') - phi(s), with phi = 0 at the game's end.
    (The research formula's LP term is left out: recipe 4's LP potential is kept as it was.)"""
    if not counts:
        return 0.0
    c = counts
    a_me = c[0] + c[1] + 0.75 * c[2] + 0.1 * min(c[3], 10)
    a_opp = c[5] + c[6] + 0.85 * c[7] + 0.1 * min(c[8], 10)
    return CA_K * max(-8.0, min(8.0, a_me - a_opp))


def practice() -> None:
    """Round 4 practice (recipe 4) from YGO_WARM, vs 0.6 WindBot / 0.2 pool / 0.2 itself, sampling at temperature 1.
    Fresh on-policy buffer: collect COLLECT finished decisions, update (<= 2 epochs, minibatch 256 as micro-batch
    accumulation, lr 1e-5, early stop when the approx-KL k3 estimate > 0.02), discard. Reward +1 win, -1 loss or
    draw, plus a constant LP potential 0.25 x (phi' - phi) per transition (Ng), minus the loop penalty.
    Critic on (YGO_VHEAD): GAE gamma=1 lambda=0.95 on V = 2 P(win) - 1, the head trained online on detached
    features; off: the running baseline. A 300-game greedy WindBot eval every 3 GPU-h and at the cap.
    Stops: cap (YGO_CAP_H); regress (two evals in a row > 0.05 below YGO_START_WR); unstable (mean clipfrac of an
    update > 0.30 after one lr halving). done-practice holds the best checkpoint among the start and the evals.
    Round 5 (YGO_R5=1): the learner plays MY_DECK, WindBot games spread over POOL_DECKS; reward win =
    max(0.85, 1 - 0.01 x turns past 5) (fast-win), loss/draw -1; plus the card-advantage potential ca_phi (Ng, with
    phi(terminal) = 0, so it can't change the best policy); optionally (YGO_RESPONSE_BONUS=1) +0.02 per own
    activation at chain link 2+ on top of an opponent's link, capped at 0.1 a game; an entropy bonus whose
    coefficient rises while the mean policy entropy is under YGO_ENT_FLOOR. Evals print the entropy too."""
    LR, EPOCHS, KL_STOP, PHI_K, LAM = 1e-5, 2, 0.02, 0.25, 0.95
    R5 = os.environ.get("YGO_R5") == "1"
    RESP = os.environ.get("YGO_RESPONSE_BONUS") == "1"
    ENT_FLOOR = float(os.environ.get("YGO_ENT_FLOOR", "0"))
    COLLECT, MINI = int(os.environ.get("YGO_COLLECT", "2048")), int(os.environ.get("YGO_MINI", "256"))  # smaller: pause test only
    shares = {"windbot": 0.6, "pool": 0.2, "learner": 0.2}
    cap_h = float(os.environ.get("YGO_CAP_H", "10"))
    start_wr = float(os.environ["YGO_START_WR"])
    eval_every_h = float(os.environ.get("YGO_EVAL_EVERY_H", "3"))
    vhead_path = os.environ.get("YGO_VHEAD", "")
    sc = Scorer(MODEL, chat=False, lora_r=LORA_R)
    model = sc.model
    sc.feature_hook()
    params = list(lora_params(model).values())
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    head = None
    vtrunk = None
    if vhead_path:
        head = ValueHead(model.config.hidden_size).to(DEVICE)
        head.load_state_dict(torch.load(vhead_path, map_location=DEVICE))
        vopt = torch.optim.Adam(head.parameters(), lr=3e-4)
        if os.environ.get("YGO_VTRUNK"):  # round 5: the critic's own LoRA trunk, kept frozen during practice
            vtrunk = torch.load(os.environ["YGO_VTRUNK"], map_location=DEVICE)
    s = {"step": 0, "gpu_s": 0.0, "evals": [], "pool": [], "games": 0, "errors": 0, "jobs": [], "seed": 40_000,
         "decisions": 0, "lr": LR, "lr_halved": False, "ret_mean": 0.0, "ret_var": 1.0, "clip_since_eval": [],
         "last_eval_gpu_s": 0.0, "critic": head is not None, "ent_coef": 0.01, "ent_since_eval": [], "r5": R5,
         "response_bonus": RESP, "ent_floor": ENT_FLOOR}
    traj: dict[str, list] = {}
    buffer: list[dict] = []
    if (CKPT / "state.json").exists():
        s.update(json.loads((CKPT / "state.json").read_text()))
        blob = torch.load(CKPT / "ckpt.pt", map_location=DEVICE, weights_only=False)
        with torch.no_grad():
            for n, p_ in lora_params(model).items():
                p_.copy_(blob["lora"][n])
        opt.load_state_dict(blob["opt"])
        buffer[:] = blob.get("buffer", [])
        traj.update(blob.get("traj", {}))
        if head is not None and "vhead" in blob:
            head.load_state_dict(blob["vhead"])
            vopt.load_state_dict(blob["vopt"])
        log("resumed", step=s["step"], msg=f"resumed step={s['step']}", gpu_h=round(s["gpu_s"] / 3600, 3))
    else:
        load_lora(sc, WARM)
        random.seed(int(os.environ.get("YGO_PRACTICE_SEED", "0")) or None)  # round 5: matched trial slices
        torch.manual_seed(int(os.environ.get("YGO_PRACTICE_SEED", "0")) or 1)
        log("fresh", warm=WARM, critic=head is not None, vtrunk=vtrunk is not None, cap_h=cap_h, start_wr=start_wr,
            r5=R5, response_bonus=RESP, ent_floor=ENT_FLOOR, deck=MY_DECK, model=MODEL)
    for g in opt.param_groups:
        g["lr"] = s["lr"]
    pool = {p_["name"]: torch.load(CKPT / "pool" / f"{p_['name']}.pt", map_location=DEVICE) for p_ in s["pool"][-POOL:]}
    if not pool:
        pool["start"] = torch.load(WARM, map_location=DEVICE)
    clock = [time.time()]

    def tick() -> None:
        now = time.time()
        s["gpu_s"] += now - clock[0]
        clock[0] = now

    def save_ckpt(reason: str) -> None:
        CKPT.mkdir(parents=True, exist_ok=True)
        blob = {"lora": {n: p_.detach() for n, p_ in lora_params(model).items()}, "opt": opt.state_dict(),
                "buffer": buffer, "traj": traj}  # kept across a pause, so the trainer resumes mid-collection
        if head is not None:
            blob.update(vhead=head.state_dict(), vopt=vopt.state_dict())
        torch.save(blob, CKPT / "ckpt.tmp")
        os.replace(CKPT / "ckpt.tmp", CKPT / "ckpt.pt")
        s["last_save_gpu_s"] = s["gpu_s"]
        save_json(CKPT / "state.json", s)
        log("checkpoint", step=s["step"], msg=f"checkpoint step={s['step']}", reason=reason)

    def snapshot() -> str:
        name = f"step-{s['step']}"
        (CKPT / "pool").mkdir(parents=True, exist_ok=True)
        sd = {n: p_.detach().clone() for n, p_ in lora_params(model).items()}
        torch.save(sd, CKPT / "pool" / f"{name}.pt")
        s["pool"].append({"name": name, "step": s["step"], "gpu_h": round(s["gpu_s"] / 3600, 3)})
        pool[name] = sd
        pool.pop("start", None) if len(pool) > POOL else None
        while len(pool) > POOL:
            del pool[next(iter(pool))]
        return name

    def play(rs, greedy, state=None):
        """Choices (and the learner's value estimates) for one batch of requests."""
        with Weights(model, state):
            fs: list = []
            with torch.no_grad():
                lg = sc.logits([r["prompt"] for r in rs], [r["n"] for r in rs], feats=fs)
                vs = [None] * len(rs)
                if head is not None and state is None and not greedy:
                    if vtrunk is not None:
                        with Weights(model, vtrunk):
                            sc.logits([r["prompt"] for r in rs], [r["n"] for r in rs], feats=fs)
                    vs = torch.sigmoid(head(torch.stack(fs))).tolist()
        out = []
        for l, v in zip(lg, vs):
            lp = torch.log_softmax(l, -1)
            a = int(torch.argmax(lp)) if greedy else int(torch.multinomial(lp.exp(), 1))
            out.append((a, float(lp[a]), v))
        return out

    def do_eval() -> float | None:
        name = f"practice-eval-step{s['step']}"
        try:
            games = int(os.environ.get("YGO_PRACTICE_EVAL_GAMES", "300"))  # fewer: smoke test only
            res = run_match(name, seats_vs("eval", "windbot", games), {"eval": lambda rs: [(a, l) for a, l, _ in play(rs, True)]},
                            DATA / "eval-state.json", seed=7000 + s["step"], on_tick=tick)
        except Stopped:
            return None
        traj.clear()  # games that ran meanwhile got stand-in answers
        if res is None:
            return None
        (DATA / "eval-state.json").unlink(missing_ok=True)
        cf = s["clip_since_eval"]
        clipfrac = round(sum(cf) / len(cf), 4) if cf else 0.0
        s["clip_since_eval"] = []
        es = s.get("ent_since_eval", [])
        entropy = round(sum(es) / len(es), 4) if es else 0.0
        s["ent_since_eval"] = []
        snap = snapshot()
        gpu_h = round(s["gpu_s"] / 3600, 2)
        s["evals"].append({"step": s["step"], "gpu_h": gpu_h, "vs_windbot": res["win_rate"], "games": res["games"],
                           "errors": res["errors"], "clipfrac": clipfrac, "entropy": entropy, "ckpt": snap,
                           "replays": res["replays"], "mean_turns": res.get("mean_turns")})
        s["last_eval_gpu_s"] = s["gpu_s"]
        log("eval", step=s["step"], gpu_h=gpu_h, vs_windbot=res["win_rate"], games=res["games"], errors=res["errors"],
            clipfrac=clipfrac, entropy=entropy,
            msg=f"eval step={s['step']} gpu_h={gpu_h} vs_windbot={res['win_rate']:.3f} games={res['games']} entropy={entropy} clipfrac={clipfrac}")
        save_ckpt("eval")
        return res["win_rate"]

    def finish(res: dict) -> None:
        s["games"] += 1
        keys = [f"{res['job']}:{res['idx']}:{k}" for k in (0, 1)]
        if "error" in res:
            s["errors"] += 1
            log("game_error", job=res["job"], idx=res["idx"], error=res["error"][:300])
            for k in keys:
                traj.pop(k, None)
            return
        for seat, k in enumerate(keys):
            ds = traj.pop(k, None)
            if not ds:
                continue
            won = res["winner"] == seat
            R = 1.0 if won else -1.0  # a draw counts as a loss
            if R5 and won:  # fast-win bonus: a loss (-1, at most -0.9 with responses) never beats a win (>= 0.85)
                R = max(0.85, 1.0 - 0.01 * max(0, res.get("turns", 0) - 5))
            phi_end = (res["lp"][seat] - res["lp"][1 - seat]) / 8000
            looped = {t for st_, t in res.get("loop_turns", []) if st_ == seat}
            phis = [d["phi"] for d in ds] + [phi_end]
            rew = [PHI_K * (phis[t + 1] - phis[t]) - (LOOP_PENALTY if ds[t].get("turn") in looped else 0.0) for t in range(len(ds))]
            if R5:  # card-advantage potential, phi(terminal) = 0
                cas = [ca_phi(d.get("counts")) for d in ds] + [0.0]
                for t in range(len(ds)):
                    rew[t] += cas[t + 1] - cas[t]
            rew[-1] += R
            if R5 and RESP:
                rew[-1] += min(0.1, 0.02 * res.get("responses", [0, 0])[seat])
            if head is not None:  # GAE, gamma 1
                V = [2 * d["v"] - 1 for d in ds] + [0.0]
                adv, run = [0.0] * len(ds), 0.0
                for t in reversed(range(len(ds))):
                    run = rew[t] + V[t + 1] - V[t] + LAM * run
                    adv[t] = run
            else:  # running baseline on the shaped return-to-go
                G, run = [0.0] * len(ds), 0.0
                for t in reversed(range(len(ds))):
                    run += rew[t]
                    G[t] = run
                for g_ in G:
                    s["ret_mean"] = 0.999 * s["ret_mean"] + 0.001 * g_
                    s["ret_var"] = 0.999 * s["ret_var"] + 0.001 * (g_ - s["ret_mean"]) ** 2
                adv = [g_ - s["ret_mean"] for g_ in G]
            for d, a_ in zip(ds, adv):
                d["adv"], d["won"] = a_, float(won)
                buffer.append(d)

    def update() -> dict:
        """<= EPOCHS passes over the buffer in minibatches of MINI; stop at approx-KL > KL_STOP. Then discard."""
        advs = torch.tensor([d["adv"] for d in buffer])
        mu, sd = float(advs.mean()), float(advs.std()) + 1e-6
        stats = {"kl": 0.0, "clipfrac": 0.0, "ent": 0.0, "vloss": 0.0, "minibatches": 0, "kl_stop": False}
        for epoch in range(EPOCHS):
            random.shuffle(buffer)
            for k in range(0, len(buffer) - MINI + 1, MINI):
                mb = buffer[k:k + MINI]
                model.train()
                opt.zero_grad(set_to_none=True)
                if head is not None and vtrunk is None:
                    vopt.zero_grad(set_to_none=True)
                kl = cf = ent = vl = 0.0
                micro = s.get("micro", MICRO)
                try:
                    for i in range(0, MINI, micro):
                        if STOP:  # a game: drop this minibatch so the GPU is free well inside the guard's 30 s
                            break
                        part = mb[i:i + micro]
                        fs: list = []
                        lg = sc.logits([d["prompt"] for d in part], [d["n"] for d in part], grad=True, feats=fs)
                        loss = 0.0
                        for d, l, f in zip(part, lg, fs):
                            lp = torch.log_softmax(l, -1)
                            logr = lp[d["a"]] - d["logp"]
                            ratio = torch.exp(logr)
                            a_ = (d["adv"] - mu) / sd
                            pg = -torch.min(ratio * a_, torch.clamp(ratio, 0.8, 1.2) * a_)
                            e = -(lp.exp() * lp).sum()
                            loss = loss + (pg - s.get("ent_coef", 0.01) * e) / MINI
                            if head is not None and vtrunk is None:
                                v = head(f.detach())
                                vloss = torch.nn.functional.binary_cross_entropy_with_logits(v, torch.tensor(d["won"], device=DEVICE))
                                loss = loss + vloss / MINI
                                vl += vloss.item() / MINI
                            r_ = ratio.item()
                            kl += ((r_ - 1) - logr.item()) / MINI  # k3: nonnegative
                            cf += float(abs(r_ - 1) > 0.2) / MINI
                            ent += e.item() / MINI
                        loss.backward()
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    model.eval()
                    torch.cuda.empty_cache()
                    s["micro"] = max(1, micro // 2)
                    log("oom_skip", step=s["step"], micro=s["micro"])
                    continue
                model.eval()
                if STOP:
                    opt.zero_grad(set_to_none=True)
                    break
                if kl > KL_STOP:  # this minibatch already drifted too far: no step, the update ends here
                    opt.zero_grad(set_to_none=True)
                    stats["kl_stop"] = True
                    log("kl_stop", step=s["step"], epoch=epoch, kl=round(kl, 5))
                    break
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if head is not None and vtrunk is None:
                    vopt.step()
                s["step"] += 1
                stats["minibatches"] += 1
                for key, v_ in (("kl", kl), ("clipfrac", cf), ("ent", ent), ("vloss", vl)):
                    stats[key] += v_
                tick()
                log("step", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), epoch=epoch, kl=round(kl, 5),
                    clipfrac=round(cf, 4), ent=round(ent, 4), vloss=round(vl, 4), lr=s["lr"])
                if STOP:
                    break
            if stats["kl_stop"] or STOP:
                break
        buffer.clear()
        m = max(1, stats["minibatches"])
        return {k: (round(v / m, 4) if k in ("kl", "clipfrac", "ent", "vloss") else v) for k, v in stats.items()}

    def submit(n: int) -> None:
        games = []
        for _ in range(n):
            u = random.random()
            opp = "windbot" if u < shares["windbot"] else ("pool:" + random.choice(list(pool))) if u < shares["windbot"] + shares["pool"] else "learner"
            seats = ["learner", opp] if random.random() < 0.5 else [opp, "learner"]
            g = {"seats": seats, "replay": random.random() < 0.002, "live": not games}
            if R5:  # the learner and its own copies play MY_DECK; WindBot plays a pool deck
                me = seats.index("learner")
                g["decks"] = [MY_DECK or None, MY_DECK or None]
                if opp == "windbot":
                    g["decks"][1 - me] = random.choice(POOL_DECKS)
            games.append(g)
        s["seed"] += 1
        job = server_call("/job", {"name": f"practice-step{s['step']}", "seed": s["seed"], "games": games})["job"]
        s["jobs"] = (s["jobs"] + [job])[-200:]

    try:
        server_call("/cancel", {})
        t_log = time.time()
        while not STOP:
            tick()
            if s["gpu_s"] >= cap_h * 3600 or s["gpu_s"] - s["last_eval_gpu_s"] >= eval_every_h * 3600:
                at_cap = s["gpu_s"] >= cap_h * 3600
                server_call("/cancel", {})
                wr = do_eval()
                if wr is None:
                    break
                ev = s["evals"]
                reason = None
                if at_cap:
                    reason = "cap"
                elif len(ev) >= 2 and all(e["vs_windbot"] < start_wr - 0.05 for e in ev[-2:]):
                    reason = "regress"
                if reason:
                    stop_practice(s, reason, start_wr)
                    return
                continue
            st = server_call("/status")
            if st["queued"] + st["running"] < LIVE_GAMES and len(buffer) < COLLECT:
                submit(LIVE_GAMES - st["queued"] - st["running"])
            nxt = server_call("/next?max=256&wait=2&gather=50", timeout=40)
            answers = []
            mine = [r for r in nxt["requests"] if r["job"] in s["jobs"]]
            answers += [{"id": r["id"], "choice": random.randrange(r["n"])} for r in nxt["requests"] if r["job"] not in s["jobs"]]
            by_tag: dict[str, list] = {}
            for r in mine:
                by_tag.setdefault(r["policy"], []).append(r)
            for tag, rs in by_tag.items():
                state = (pool.get(tag[5:]) or next(iter(pool.values()))) if tag.startswith("pool:") else None
                for r, (a, lp, v) in zip(rs, play(rs, greedy=False, state=state)):
                    answers.append({"id": r["id"], "choice": a})
                    if tag == "learner":
                        traj.setdefault(f"{r['job']}:{r['idx']}:{r['seat']}", []).append(
                            {"prompt": r["prompt"], "n": r["n"], "a": a, "logp": lp, "v": v,
                             "phi": (r["lp"][0] - r["lp"][1]) / 8000, "turn": r.get("turn"), "counts": r.get("counts")})
                        s["decisions"] += 1
            if answers:
                server_call("/answer", answers)
            for res in nxt["results"]:
                if res["job"] in s["jobs"]:
                    finish(res)
            if len(buffer) >= COLLECT and not STOP:
                stats = update()
                s["clip_since_eval"].append(stats["clipfrac"])
                s.setdefault("ent_since_eval", []).append(stats["ent"])
                if ENT_FLOOR and stats["minibatches"]:  # exploration: keep the policy's entropy above the floor
                    c = s.get("ent_coef", 0.01)
                    c = min(0.2, c * 1.5) if stats["ent"] < ENT_FLOOR else max(0.01, c / 1.5) if stats["ent"] > 2 * ENT_FLOOR else c
                    s["ent_coef"] = round(c, 5)
                    stats["ent_coef"] = s["ent_coef"]
                log("update", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), games=s["games"], **stats)
                if stats["clipfrac"] > 0.30:
                    if s["lr_halved"]:
                        server_call("/cancel", {})
                        wr = do_eval()
                        if wr is None:
                            break
                        stop_practice(s, "unstable", start_wr)
                        return
                    s["lr"] /= 2
                    s["lr_halved"] = True
                    for g in opt.param_groups:
                        g["lr"] = s["lr"]
                    log("lr_halved", step=s["step"], lr=s["lr"], clipfrac=stats["clipfrac"])
                save_ckpt("update")
            if time.time() - t_log > 300:
                t_log = time.time()
                active = set(int(j) for j in server_call("/status").get("active", {}))
                for k in [k for k in traj if int(k.split(":")[0]) not in active]:
                    del traj[k]
                log("progress", step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 3), games=s["games"], errors=s["errors"],
                    decisions=s["decisions"], buffer=len(buffer))
    except Stopped:
        pass
    if not (DATA / "done-practice").exists():
        save_ckpt("stop")  # games in flight wait for answers on the server; their trajectories resume with us


def stop_practice(s: dict, reason: str, start_wr: float) -> None:
    server_call("/cancel", {})
    ev = s["evals"]
    best = max(ev, key=lambda e: e["vs_windbot"]) if ev else None
    pick = {"ckpt": str(CKPT / "pool" / f"{best['ckpt']}.pt"), "step": best["step"], "vs_windbot": best["vs_windbot"]} \
        if best and best["vs_windbot"] > start_wr else {"ckpt": WARM, "step": "start", "vs_windbot": start_wr}
    out = {"reason": reason, "gpu_s": round(s["gpu_s"], 1), "evals": ev, "best": pick, "start_wr": start_wr}
    save_json(DATA / "result.json", out)
    log("train_stop", reason=reason, step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 2), best=pick,
        msg=f"train_stop reason={reason}")
    (DATA / "done-practice").write_text(reason)


# ------------------------------------------------------------------ zeroshot

def zeroshot_players():
    sc = Scorer(ZEROSHOT_MODEL, chat=True, budget=int(os.environ.get("YGO_ZEROSHOT_BUDGET", "2048")))
    log("zeroshot_loaded", model=ZEROSHOT_MODEL, vram_mib=torch.cuda.memory_allocated() // 2**20 if DEVICE == "cuda" else 0)
    return sc, (lambda rs: choose(sc, rs, greedy=True))


def zeroshot() -> None:
    games = int(os.environ.get("YGO_ZEROSHOT_GAMES", "200"))
    opp = os.environ.get("YGO_ZEROSHOT_OPP", "random")  # "windbot": the fallback-teacher test
    _, play = zeroshot_players()
    tag = "" if opp == "random" else f"-vs-{opp}"
    picks_log = os.environ.get("YGO_PICKS_LOG", "")  # round 3: ZeroShot's picks as teacher labels (teacher-pick format)
    if picks_log:
        inner = play

        def play(rs):
            out = inner(rs)
            with open(picks_log, "a") as f:
                for r, (a, _) in zip(rs, out):
                    f.write(json.dumps({"job": r["job"], "idx": r["idx"], "seat": r["seat"], "turn": r.get("turn"),
                                        "prompt": r["prompt"], "n": r["n"], "choice": a}) + "\n")
            return out
    res = run_match(f"zeroshot-{games}{tag}", seats_vs("zeroshot", opp, games), {"zeroshot": play}, DATA / f"zeroshot{tag}-state.json",
                    seed=7)
    if res is None:
        return
    save_json(DATA / f"zeroshot{tag}.json", {"model": ZEROSHOT_MODEL, "opp": opp, **res})
    log("zeroshot_done", model=ZEROSHOT_MODEL, opp=opp, games=res["games"], errors=res["errors"], win_rate=res["win_rate"],
        loop_breaks=res["loop_breaks"],
        msg=f"zeroshot_model={ZEROSHOT_MODEL} opp={opp} zeroshot_games={res['games']} errors={res['errors']} win_rate={res['win_rate']:.3f}")
    (DATA / "done-zeroshot").write_text("ok")


# ------------------------------------------------------------------ table

def table() -> None:
    st = json.loads((CKPT / "state.json").read_text())
    hist = st["pool"]  # every checkpoint ever saved, oldest first; files kept in ckpt/pool
    if len(hist) <= 5:
        picks = list(hist)
    else:
        picks = [hist[round(k * (len(hist) - 1) / 4)] for k in range(5)]
    sc = Scorer(MODEL, chat=False, lora_r=st["recipe"]["lora_r"])
    load = lambda n: torch.load(CKPT / "pool" / f"{n}.pt", map_location=DEVICE)
    states = {p["name"]: load(p["name"]) for p in picks}
    prev = {}
    for k, p in enumerate(picks):
        if k > 0:
            prev[p["name"]] = picks[k - 1]["name"]
        else:
            i = next(j for j, h in enumerate(hist) if h["name"] == p["name"])
            prev[p["name"]] = hist[i - 1]["name"] if i > 0 else "base"
            if prev[p["name"]] != "base":
                states[prev[p["name"]]] = load(prev[p["name"]])
    zero = {n: torch.zeros_like(t) for n, t in next(iter(states.values())).items() if "lora_B" in n}
    base_state = {n: (zero[n] if n in zero else t) for n, t in next(iter(states.values())).items()}  # B=0: no adapter
    states["base"] = base_state
    opps = os.environ.get("YGO_TABLE_OPPS", "windbot,zeroshot,random,prev").split(",")
    ref = os.environ.get("YGO_TABLE_REF", "")  # round 3: step-1239, the round-1 model both rounds warm-started from
    if "ref" in opps:
        states["ref"] = torch.load(ref, map_location=DEVICE)
    zeroshot_play = zeroshot_players()[1] if "zeroshot" in opps else None
    out = DATA / "table.json"
    rows = json.loads(out.read_text()) if out.exists() else {}
    for p in picks:
        for opp in opps:
            key = f"{p['name']}:{opp}"
            if key in rows:
                continue
            opp_tag = prev[p["name"]] if opp == "prev" else opp
            me = f"ck:{p['name']}"
            players = {me: (lambda rs, n=p["name"]: choose(sc, rs, greedy=True, state=states[n]))}
            if opp == "zeroshot":
                players["zeroshot"] = zeroshot_play
            elif opp == "prev":
                opp_tag = f"ck:{opp_tag}"
                players[opp_tag] = (lambda rs, n=prev[p["name"]]: choose(sc, rs, greedy=True, state=states[n]))
            elif opp == "ref":
                opp_tag = "ck:ref"
                players[opp_tag] = (lambda rs: choose(sc, rs, greedy=True, state=states["ref"]))
            res = run_match(f"table-{p['name']}-vs-{opp}", seats_vs(me, opp_tag, 100), players,
                            DATA / "table-state.json", seed=500 + p["step"], replay_first=1)
            if res is None:
                return
            (DATA / "table-state.json").unlink(missing_ok=True)
            rows[key] = {"ckpt": p["name"], "step": p["step"], "gpu_h": p["gpu_h"], "opp": opp,
                         "opp_name": prev[p["name"]] if opp == "prev" else Path(ref).stem if opp == "ref" else opp, **res}
            save_json(out, rows)
            log("table_row", **{k: rows[key][k] for k in ("ckpt", "opp", "opp_name", "games", "win_rate", "errors")})
    head = {"windbot": "vs_windbot", "zeroshot": "vs_zeroshot", "random": "vs_random", "prev": "vs_previous",
            "ref": f"vs_{Path(ref).stem}" if ref else "vs_ref"}
    lines = [f"{'checkpoint':<12} {'step':>6} {'gpu_h':>6}  " + "  ".join(f"{head[o]:>12}" for o in opps) + "  errors"]
    for p in picks:
        r = {o: rows[f"{p['name']}:{o}"] for o in opps}
        err = sum(x["errors"] for x in r.values())
        cells = [f"{r[o]['win_rate']:>5.2f} ({r[o]['opp_name']})" if o == "prev" else f"{r[o]['win_rate']:>12.2f}" for o in opps]
        lines.append(f"{p['name']:<12} {p['step']:>6} {p['gpu_h']:>6}  " + "  ".join(cells) + f"  {err}")
    total_games = sum(x["games"] for x in rows.values())
    lines.append(f"table_games={total_games} errors={sum(x['errors'] for x in rows.values())} "
                 f"(each row: 100 games per opponent, greedy, half as first player; draws count as non-wins)")
    (DATA / "table.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log("table", msg=l)
    (DATA / "done-table").write_text("ok")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    {"train": train, "zeroshot": zeroshot, "table": table, "textcheck": textcheck, "match": match, "sl": sl,
     "values": values, "values2": values2, "practice": practice}[MODE]()

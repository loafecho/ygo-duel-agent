"""ygo-worker: a plain venv process on the GPU workstation's 3060 (~/.local/share/ygo-worker/venv), started by supervisor.py.
It connects OUT to the game server on the game host (ygo-runner serve, :8765). Nothing listens on the GPU workstation.

Every decision arrives as a prompt (board, decision, lettered options, "Answer:") built by the runner.
A model scores it with its first-token logits over the option letters: one forward pass per decision.

Modes:
  train  self-play PPO on Qwen2.5-0.5B + LoRA. Opponents: itself, a pool of its last 5 checkpoints, and 20% random.
         Reward +1 win, -1 loss or draw, plus LP-difference shaping that fades to 0 by half the GPU-hour cap.
         A 200-game greedy eval vs random runs at the start and every EVAL_EVERY GPU-hours; training stops
         when one reaches TARGET or the cap is spent ("train_stop reason=target|cap").
  zeroshot  the ZeroShot baseline: Qwen3.5-4B, no training, 200 games vs random.
  table  Phase 3: the final checkpoint and up to 4 earlier ones, 100 games each vs random, ZeroShot, the previous one.

SIGTERM (the supervisor's game guard) makes every mode save its state and exit so the GPU is freed;
the next start resumes. Games on the game host just wait for answers meanwhile.
Logs: one JSON object per line on stdout and <YGO_DATA>/logs/<mode>.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import random
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
TARGET = 0.60
EVAL_EVERY_H = float(os.environ.get("YGO_EVAL_EVERY_H", "2.0"))
EVAL_GAMES = int(os.environ.get("YGO_EVAL_GAMES", "200"))
CKPT_EVERY_S = 1800    # a pool snapshot (opponent + Phase 3 table candidate)
SAVE_EVERY_S = 600     # the resumable checkpoint, so a crash costs at most this much
POOL = 5
BATCH = int(os.environ.get("YGO_BATCH", "32"))
MICRO = int(os.environ.get("YGO_MICRO", "4"))
BUFFER_CAP = 4096
KEEP_P = float(os.environ.get("YGO_KEEP_P", "1.0"))         # share of a finished game's decisions kept (was 0.35 in run 1: the buffer starved, so keep all)
PPO_EPOCHS = int(os.environ.get("YGO_PPO_EPOCHS", "2"))  # each decision is used for this many updates
LIVE_GAMES = int(os.environ.get("YGO_LIVE_GAMES", "96"))
MAX_TOKENS = 1536
STOP = False
MODE = sys.argv[1] if len(sys.argv) > 1 else ""


def recipe() -> dict:
    r = {"lr": 2e-5, "clip": 0.2, "entropy": 0.01, "shaping": 0.5, "lora_r": 16, "temp": 1.0, "random_share": 0.2,
         "pool_share": 0.4}
    if os.environ.get("YGO_RECIPE") == "2":
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

    def format(self, prompt: str) -> str:
        if not self.chat:
            return prompt
        body = prompt.removesuffix("Answer:") + "Reply with only the letter of the best option."
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def encode(self, prompts: list[str]) -> list[list[int]]:
        out = []
        for p in prompts:
            ids = self.tok(self.format(p), add_special_tokens=False).input_ids
            out.append(ids[-MAX_TOKENS:])  # keep the end: the options and "Answer:" must survive
        return out

    def logits(self, prompts: list[str], ns: list[int], grad: bool = False) -> list[torch.Tensor]:
        """Letter logits (float32, length n_i) for each prompt, in input order."""
        enc = self.encode(prompts)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out: list = [None] * len(enc)
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
                    p.copy_(self.state[n])

    def __exit__(self, *a):
        if self.state is not None:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(self.saved[n])


def choose(sc: Scorer, reqs: list[dict], greedy: bool, temp: float = 1.0, state=None):
    """(choice, logp of choice) per request."""
    if not reqs:
        return []
    with Weights(sc.model, state):
        lg = sc.logits([r["prompt"] for r in reqs], [r["n"] for r in reqs])
    out = []
    for l in lg:
        lp = torch.log_softmax(l / temp, -1)
        a = int(torch.argmax(lp)) if greedy else int(torch.multinomial(lp.exp(), 1))
        out.append((a, float(lp[a])))
    return out


# ------------------------------------------------------------------ matches (eval, zeroshot, table)

def seats_vs(tag: str, opp: str, games: int) -> list[dict]:
    """Half the games with `tag` as player 0 (goes first), half as player 1."""
    return [{"seats": [tag, opp] if g % 2 == 0 else [opp, tag]} for g in range(games)]


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
        seat = res["seats"].index(tag)
        if res["winner"] == seat:
            wins += 1
        elif res["winner"] == -1:
            draws += 1
        else:
            losses += 1
    n = len(st["results"])
    replays = [r["replay"] for r in st["results"].values() if r.get("replay")]
    loop_breaks = sum(r.get("loop_breaks", 0) for r in st["results"].values())
    return {"name": st["name"], "tag": tag, "games": n, "loop_breaks": loop_breaks, "wins": wins, "losses": losses, "draws": draws, "errors": errors,
            "win_rate": round(wins / max(n, 1), 3), "replays": replays,
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
        log("fresh", step=0, model=MODEL, recipe=R)
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
        try:
            res = run_match(f"eval-step{s['step']}", seats_vs("eval", "random", EVAL_GAMES),
                            {"eval": lambda rs: choose(sc, rs, greedy=True)}, DATA / "eval-state.json",
                        seed=1000 + s["step"], on_tick=tick)
        except Stopped:
            return None
        traj.clear()  # games that ran during the eval got stand-in answers; their trajectories are incomplete
        if res is None:
            return None
        (DATA / "eval-state.json").unlink(missing_ok=True)
        gpu_h = round(s["gpu_s"] / 3600, 2)
        s["evals"].append({"step": s["step"], "gpu_h": gpu_h, "vs_random": res["win_rate"], "games": res["games"],
                           "errors": res["errors"], "replays": res["replays"]})
        log("eval", step=s["step"], gpu_h=gpu_h, vs_random=res["win_rate"], games=res["games"], errors=res["errors"],
            wins=res["wins"], draws=res["draws"], loop_breaks=res["loop_breaks"],
            msg=f"eval step={s['step']} gpu_h={gpu_h} vs_random={res['win_rate']:.3f} games={res['games']}")
        save_json(CKPT / "state.json", s)
        return res["win_rate"]

    def submit_games(n: int) -> None:
        games = []
        for _ in range(n):
            u = random.random()
            if u < R["random_share"]:
                opp = "random"
            elif u < R["random_share"] + R["pool_share"] and pool:
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
            for d in ds:
                if random.random() < KEEP_P:
                    d["ret"] = R_ + c * (phi_end - d["phi"])
                    buffer.append(d)
        del buffer[: max(0, len(buffer) - BUFFER_CAP)]

    def update() -> dict | None:
        try:
            return update_once()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            model.eval()
            torch.cuda.empty_cache()
            log("oom_skip", step=s["step"])  # this batch is dropped; the next one tries again
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
        for i in range(0, BATCH, MICRO):
            mb = batch[i:i + MICRO]
            lg = sc.logits([d["prompt"] for d in mb], [d["n"] for d in mb], grad=True)
            loss = 0.0
            for d, l, a in zip(mb, lg, adv[i:i + MICRO]):
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
                        {"prompt": r["prompt"], "n": r["n"], "a": a, "logp": lp, "phi": (r["lp"][0] - r["lp"][1]) / 8000})
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
        done = s["gpu_s"] >= CAP_H * 3600
        if (s["gpu_s"] - last_eval_gpu_s >= EVAL_EVERY_H * 3600 or done) and EVAL_GAMES > 0 and not STOP:
            server_call("/cancel", {})
            wr = do_eval()
            if wr is None:
                break
            last_eval_gpu_s = s["gpu_s"]
            add_pool()
            save_ckpt("eval")
            if wr >= TARGET or done:
                reason = "target" if wr >= TARGET else "cap"
                server_call("/cancel", {})
                log("train_stop", reason=reason, step=s["step"], gpu_h=round(s["gpu_s"] / 3600, 2), vs_random=wr,
                    msg=f"train_stop reason={reason}")
                (DATA / "done-train").write_text(reason)
                return


# ------------------------------------------------------------------ zeroshot

def zeroshot_players():
    sc = Scorer(ZEROSHOT_MODEL, chat=True, budget=int(os.environ.get("YGO_ZEROSHOT_BUDGET", "2048")))
    log("zeroshot_loaded", model=ZEROSHOT_MODEL, vram_mib=torch.cuda.memory_allocated() // 2**20 if DEVICE == "cuda" else 0)
    return sc, (lambda rs: choose(sc, rs, greedy=True))


def zeroshot() -> None:
    games = int(os.environ.get("YGO_ZEROSHOT_GAMES", "200"))
    _, play = zeroshot_players()
    res = run_match(f"zeroshot-{games}", seats_vs("zeroshot", "random", games), {"zeroshot": play}, DATA / "zeroshot-state.json", seed=7)
    if res is None:
        return
    save_json(DATA / "zeroshot.json", {"model": ZEROSHOT_MODEL, **res})
    log("zeroshot_done", model=ZEROSHOT_MODEL, games=res["games"], errors=res["errors"], win_rate=res["win_rate"],
        loop_breaks=res["loop_breaks"],
        msg=f"zeroshot_model={ZEROSHOT_MODEL} zeroshot_games={res['games']} errors={res['errors']} win_rate={res['win_rate']:.3f}")
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
    zeroshot_sc, zeroshot_play = zeroshot_players()
    out = DATA / "table.json"
    rows = json.loads(out.read_text()) if out.exists() else {}
    for p in picks:
        for opp in ("random", "zeroshot", "prev"):
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
            res = run_match(f"table-{p['name']}-vs-{opp}", seats_vs(me, opp_tag, 100), players,
                            DATA / "table-state.json", seed=500 + p["step"], replay_first=1)
            if res is None:
                return
            (DATA / "table-state.json").unlink(missing_ok=True)
            rows[key] = {"ckpt": p["name"], "step": p["step"], "gpu_h": p["gpu_h"], "opp": opp,
                         "opp_name": prev[p["name"]] if opp == "prev" else opp, **res}
            save_json(out, rows)
            log("table_row", **{k: rows[key][k] for k in ("ckpt", "opp", "opp_name", "games", "win_rate", "errors")})
    lines = [f"{'checkpoint':<12} {'step':>6} {'gpu_h':>6}  {'vs_random':>9}  {'vs_zeroshot':>8}  {'vs_previous':>18}  errors"]
    for p in picks:
        r = {o: rows[f"{p['name']}:{o}"] for o in ("random", "zeroshot", "prev")}
        err = sum(x["errors"] for x in r.values())
        lines.append(f"{p['name']:<12} {p['step']:>6} {p['gpu_h']:>6}  {r['random']['win_rate']:>9.2f}  "
                     f"{r['zeroshot']['win_rate']:>8.2f}  {r['prev']['win_rate']:>5.2f} ({r['prev']['opp_name']:>10})  {err}")
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
    {"train": train, "zeroshot": zeroshot, "table": table}[MODE]()

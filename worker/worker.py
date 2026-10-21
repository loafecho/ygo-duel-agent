"""ygo-worker: a plain venv process on the GPU workstation's 3060 (~/.local/share/ygo-worker/venv), started by supervisor.py.

Two modes:
  serve  connects OUT to the duel broker on the game host (long-poll), scores each legal option with
         Qwen2.5-0.5B (mean log-prob of the option text after the prompt), answers the argmax.
         Nothing listens on the GPU workstation.
  train  a smoke LoRA training loop. SIGTERM (sent by the supervisor when a game
         starts) finishes the current step, writes a checkpoint and exits so the GPU is freed.
         On the next start it resumes from that checkpoint.

Logs are one JSON object per line on stdout and <YGO_DATA>/logs/<mode>.jsonl.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

import torch
from torch import no_grad
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("YGO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BROKER = os.environ.get("YGO_BROKER", "http://198.51.100.10:8765")
DATA = Path(os.environ.get("YGO_DATA", Path.home() / ".local/share/ygo-worker/data"))
CKPT = DATA / "ckpt"
STOP = False


def log(mode: str, event: str, **kw) -> None:
    line = json.dumps({"t": round(time.time(), 3), "event": event, **kw})
    print(line, flush=True)
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    with open(DATA / "logs" / f"{mode}.jsonl", "a") as f:
        f.write(line + "\n")


def on_term(signum, frame) -> None:
    global STOP
    STOP = True


def load(dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to("cuda")
    return tok, model


# ------------------------------------------------------------------ serve

@no_grad()
def score(tok, model, prompt: str, options: list[str]) -> list[float]:
    p_ids = tok(prompt, return_tensors="pt").input_ids[0]
    seqs = [torch.cat([p_ids, tok(" " + o, return_tensors="pt").input_ids[0]]) for o in options]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), width), pad)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
    out = []
    for i, s in enumerate(seqs):
        n = len(s) - len(p_ids)
        tgt = ids[i, len(p_ids) : len(s)]
        lp = logp[i, len(p_ids) - 1 : len(s) - 1].gather(1, tgt[:, None]).squeeze(1)
        out.append(lp.sum().item() / max(n, 1))
    return out


def serve() -> None:
    tok, model = load()
    log("serve", "loaded", model=MODEL, vram_mib=torch.cuda.memory_allocated() // 2**20)
    while not STOP:
        try:
            with urllib.request.urlopen(f"{BROKER}/next", timeout=35) as r:
                if r.status == 204:
                    continue
                req = json.load(r)
        except Exception as e:  # broker down or between runs
            log("serve", "broker_wait", err=type(e).__name__)
            time.sleep(3)
            continue
        t0 = time.time()
        scores = score(tok, model, req["prompt"], req["options"])
        choice = max(range(len(scores)), key=scores.__getitem__)
        body = json.dumps({"id": req["id"], "choice": choice}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{BROKER}/answer", body, {"Content-Type": "application/json"}), timeout=10).read()
        log("serve", "request", id=req["id"], game=req.get("game"), n_options=len(scores), choice=choice, ms=round((time.time() - t0) * 1000))
    log("serve", "stopped")


# ------------------------------------------------------------------ train (smoke)

SMOKE = [
    "Duel state: my LP 8000, opponent LP 8000, my hand has 5 cards. Best action: summon a monster.",
    "Duel state: opponent has no monsters, I control Cyber Dragon. Best action: attack directly.",
    "Duel state: opponent controls a face-down card. Best action: set a trap and pass.",
    "Duel state: my LP 1200, opponent attacks. Best action: activate the trap card.",
]


def train() -> None:
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict

    tok, base = load(dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    step = 0
    if (CKPT / "state.json").exists():
        st = torch.load(CKPT / "ckpt.pt", map_location="cuda")
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"])
        step = json.loads((CKPT / "state.json").read_text())["step"]
        log("train", "resumed", step=step, msg=f"resumed step={step}")
    else:
        log("train", "fresh", step=0)
    batch = tok(SMOKE, return_tensors="pt", padding=True).to("cuda")
    labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
    while not STOP:
        loss = model(**batch, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log("train", "step", step=step, loss=round(loss.item(), 4))
        time.sleep(0.5)  # smoke loop: keep it gentle, ~2 steps/s
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict()}, CKPT / "ckpt.tmp")
    os.replace(CKPT / "ckpt.tmp", CKPT / "ckpt.pt")
    (CKPT / "state.json").write_text(json.dumps({"step": step}))
    log("train", "checkpoint", step=step, msg=f"checkpoint step={step}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    {"serve": serve, "train": train}[sys.argv[1]]()

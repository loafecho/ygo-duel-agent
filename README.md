# ygo-duel-agent

A small language model that learns to play Yu-Gi-Oh, plus everything around it: a Rust duel runner on the
open-source EDOPro engine, a game server that hands decisions to a GPU worker over HTTP, a Python trainer
(self-play PPO, imitation of a scripted bot, DAgger), a guard that gives the GPU back whenever someone starts a
game on the same machine, and a browser replay viewer.

## Why it exists

It started as a question: can a 0.5B-1.5B model on one consumer GPU (an RTX 3060, 12 GB) learn a real card game
from text prompts alone, inside a fixed compute budget, with results that hold up? The project is built around
that last part. Each training round was planned before it ran, evaluations are greedy and fixed in advance,
changes had to beat the best model so far on a pre-registered match, and the checks that keep the model honest
(no hidden information in its prompts, deterministic options, a verified pause on the shared GPU) are code in
this repository rather than claims.

## How it works

```
 game host (CPU only)                                  GPU workstation
 ┌──────────────────────────────────────────┐          ┌──────────────────────────────────────┐
 │ ygo-runner serve  (Rust, ygopro-core FFI) │  HTTP    │ supervisor.py  (game guard)          │
 │   runs up to N duels in parallel          │◄─────────┤   └─ worker.py <mode>                │
 │   random / WindBot seats played locally   │ long-poll│        Qwen2.5 + LoRA, scores the    │
 │   other seats: prompt -> /next -> /answer │          │        option letters, trains        │
 │ WindBot server (scripted bot, .NET)       │          └──────────────────────────────────────┘
 │ viewer/server.py  replays + live board    │
 └──────────────────────────────────────────┘
```

**Runner (`runner/`, Rust).** Links EDOPro's engine ([edo9300/ygopro-core](https://github.com/edo9300/ygopro-core))
through its own FFI (`ffi.rs`), loads card data from ProjectIgnis BabelCDB with rusqlite and card scripts from
ProjectIgnis CardScripts (`data.rs`). `decide.rs` turns every message the engine waits on into a list of legal
choices that carry the exact response bytes; combinatorial selections are offered as up to 16 sampled legal
answers, seeded per message so a retry sees the same options. `board.rs` reads the zones, and the same snapshot
feeds the model prompt (the opponent's hidden cards hidden) and the replay file (everything). `history.rs` adds a
whitelist of recent public plays. `windbot.rs` lets ProjectIgnis WindBot play a seat by acting as the EDOPro host
on a loopback port.

Subcommands: `random` (random vs random, optional replays), `prompt` (print one real prompt), `serve` (the game
server, `broker.rs`), `optcheck` (sampled options re-parse identically), `leakcheck` (no hidden opponent card in a
player's prompts), `ceiling` (WindBot vs WindBot per deck).

**Game server protocol (`broker.rs`).** The worker connects out; nothing on the GPU machine listens.
`POST /job` queues games with a policy tag per seat and returns a job token, `GET /next` long-polls for waiting
decisions and the finished results of the caller's job token, `POST /answer` returns choices under the lease each
decision was handed out with (unknown, stale or repeated answers get a 409). A request handed out but unanswered
for 90 s is handed out again under a new lease, so a worker that was stopped for a game loses nothing; a duel
waiting on the worker simply waits.

**Worker (`worker/worker.py`).** Each decision is one forward pass: the prompt ends in a lettered option list and
`Answer:`, and the policy is the softmax of the first-token logits over the option letters. Modes:

| Mode | What it does |
|---|---|
| `train` | self-play PPO with LoRA, opponents mixed from WindBot, itself, a pool of past checkpoints and random; optional imitation warm-up from labels |
| `zeroshot` | an untrained 4B model as a baseline seat |
| `table` | final checkpoint and earlier ones against each opponent |
| `textcheck` | which card-text form (full, one-line, none) best predicts WindBot's picks on held-out games |
| `sl` | supervised imitation on WindBot and DAgger labels |
| `match` | a pre-registered greedy match with a Wilson 95% interval, paired seeds, optional shadow-WindBot labelling (DAgger) |
| `values`, `values2` | a win-probability critic on the policy's hidden state |
| `practice` | on-policy PPO from a warm start against WindBot, the checkpoint pool and itself, with potential-based shaping, an optional critic (GAE) and stop rules for regressions and instability |

**Supervisor (`worker/supervisor.py`).** Owns the worker process. Every 2 s it checks for a game (Steam's
launcher, gamescope wrapping a command, another GPU job named by `YGO_YIELD_TO`, or other GPU load above 50%).
On a game it sends SIGTERM; the worker writes a resumable checkpoint and exits so its VRAM is released, and
after two clean checks it is started again from the same step. `worker/pause_test.sh` proves this end to end with
a fake game: GPU freed within 10 s, VRAM back within 300 MiB of the baseline, resumed within 5 min, 10 more steps.

**Pipelines (`worker/pipeline*.sh`).** One script per training round (phases 1-3, then rounds 2-5). Each stage
runs under the supervisor, leaves a `stage-*.done` marker so a rerun skips it, and waits for marker files from
steps that run on the game host. Rounds 4 and 5 keep a GPU-hour ledger and a reserve for the final match.

**Tools (`tools/`).** Label builders for imitation and DAgger, a WindBot sanity check, the card-summary checker,
and `export_hf.py`, which exports a LoRA checkpoint as a Hugging Face adapter and verifies the export by reloading
it and comparing tensors and letter logits. `deploy/hf/` holds the model card and an upload script that reads the
token from `HF_TOKEN` or `$HF_HOME/token` and never prints it.

## Results recorded for the final adapter

From `deploy/hf/README.md` (greedy play, a draw counts as a loss; the run logs are not in this repository):
0.936 [0.911, 0.954] against WindBot over 500 games with Qwen2.5-1.5B + LoRA on the Kashtira deck, and 0.610
pooled against WindBot piloting five other decks. Most of the gain came from switching decks; the step from 0.5B
to 1.5B added 0.004.

## Running it

What is not in this repository and has to be fetched: the engine source, ProjectIgnis CardScripts and BabelCDB,
WindBot and its decks, and model weights from Hugging Face. Card names and text belong to Konami and are not
shipped here.

Game host (Linux, CPU only):

1. Build edo9300/ygopro-core as a shared library and set `OCGCORE_DIR` to the directory holding `libocgcore.so`.
2. Clone ProjectIgnis CardScripts and BabelCDB; pass them with `--scripts` and `--cdb` (defaults live under `/srv/ygo/`).
3. `cd runner && cargo build --release`, then `ygo-runner random --games 100` to check the engine works.
4. Optional: build ProjectIgnis WindBot and run it in ServerMode (`deploy/ygo-windbot.service`).
5. Start `ygo-runner serve` and `viewer/server.py` (`deploy/ygo-server.service`, `deploy/ygo-viewer.service`).

GPU workstation:

1. Create a venv with torch, transformers and peft (the versions used are in `worker/pipeline.sh`) and copy
   `worker/` to `~/.local/share/ygo-worker/app/`.
2. Point the worker at the game server: `export YGO_BROKER=http://game-host.example:8765`.
3. Run a mode under the guard: `worker/supervisor.py train` (logs in `~/.local/share/ygo-worker/logs/`).

`YGO_CARD_TEXT=oneliner` reads a JSON file of one-line card summaries (`YGO_ONELINERS`, a `{"card name":
"summary"}` object next to `worker.py`). Those files are not shipped, because they paraphrase copyrighted card
text. To make one, write a summary for every card in your decks, then run
`tools/oneliners_check.py --deck <deck.ydk> --cdb <cards.cdb> --file <summaries.json>`, which lists any card that
is missing. The default `YGO_CARD_TEXT=full` uses the card text from the CDB and needs no extra file.

## Tests and checks

```
pip install pytest==9.0.3 ruff==0.15.15
pytest -q                     # worker helpers, the supervisor's game guard, the viewer over HTTP
ruff check .
shellcheck -S warning $(git ls-files '*.sh')
cd runner && RUSTFLAGS="-D warnings" cargo check --locked
```

CI (`.github/workflows/ci.yml`) runs all four. The Python tests run without a GPU or torch (`tests/conftest.py`
registers a stand-in when torch is missing); the supervisor tests start real processes (one whose program name is
`gamescope`, one whose script path matches `YGO_YIELD_TO`) and check the guard sees them. The Rust job type-checks the runner but does not link or run it,
because that needs the engine library.

## Limits

- The runner has no unit tests of its own; its correctness checks (`optcheck`, `leakcheck`, `random`) need the
  engine and card data, so they run on a game host, not in CI.
- The pipelines are records of how each round was run on one specific two-machine setup. They assume the
  `~/.local/share/ygo-worker` layout and marker files from the game host, and are not a general training CLI.
- The model plays one deck well and learned mostly by copying a scripted bot, so its strength is measured against
  that bot, not against people.
- The pause guarantee was measured on one machine (`worker/pause_test.sh`); the guard detects Steam and gamescope
  launches, not every possible game.

## License

MIT, see `LICENSE`. EDOPro's engine, ProjectIgnis data, WindBot and the Qwen models have their own licenses.

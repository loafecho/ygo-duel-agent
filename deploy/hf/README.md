---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
tags:
- lora
- peft
- game-ai
- imitation-learning
- dagger
- yu-gi-oh
---

# Yu-Gi-Oh agent: Kashtira, LoRA on Qwen2.5-1.5B

A LoRA adapter (rank 16, about 74 MB) that turns [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) into a Yu-Gi-Oh duel agent. It plays the Kashtira deck in the open-source EDOPro / ygopro-core engine. This is a fine-tune of Qwen's model, not a model trained from scratch.

## How it plays

At each decision the engine's legal moves are written out as a lettered list (A, B, C, ...) under a text description of the public board: life points, hand, field, graveyards, and short one-line card summaries. The model reads that prompt and scores the next token. The highest-scoring option letter is the move. Every evaluation below picks greedily, with no sampling and no search.

## Training

Everything ran on one RTX 3060 (12 GB) under a 48 GPU-hour budget, with the engine on a separate CPU-only box.

1. **Imitation.** Supervised training on the moves WindBot (the engine's scripted bot) makes when it plays Kashtira. Held-out move accuracy is 0.903.
2. **DAgger.** The model played its own games, WindBot labelled the positions it reached, and those corrections were added to the training set.
3. **Gated changes.** Every change had to beat the best model so far on a 500-game evaluation, under rules fixed before the run. The deck switch (from Cyber Dragon to Kashtira) and the jump from 0.5B to 1.5B both passed.

A separate value head (AUC 0.85 at predicting the winner) was trained but is not part of this adapter.

## Results (greedy, 95% Wilson intervals)

| Opponent | Win rate | Games |
|---|---|---|
| WindBot (Cyber Dragon) | **0.936** [0.911, 0.954] | 500 |
| The previous round's agent (0.5B) | **0.940** [0.907, 0.962] | 300 |
| WindBot, 5 other decks (pooled) | 0.610 | 300 |

By deck: Blackwing 0.97, Dragunity 0.90, Zoodiac 0.47, Trickstar 0.42, Thunder Dragon 0.30. A draw counts as a loss.

**What drove the gain.** The start of this round scored 0.486 against WindBot. Most of the climb came from the deck: WindBot itself wins only 10% with the old Cyber Dragon deck and 77% with Kashtira. Moving the 0.5B model to Kashtira reached 0.918. The 1.5B base added only 0.004 more, at about 2.6× the compute per move.

## Limits

- **One deck.** It only plays Kashtira and has seen only WindBot-style opponents. It is clearly weaker against decks it never trained against.
- **WindBot-level play.** It learned mostly by copying a scripted bot, so it isn't a human-level player, and the 0.936 measures it against that bot.
- **Needs the runner.** The prompts come from a custom duel runner plus a separately fetched engine, card scripts and card database, so the adapter can't play a duel by itself.
- **No card data here.** Card names and text belong to Konami. This repo holds only model weights and a description.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "<this repo>")
tok = AutoTokenizer.from_pretrained("<this repo>")
# Prompt: board description + "Options:\nA. ...\nB. ...\nAnswer:"; take the logits of " A", " B", ... at the last position.
```

Built by loafecho, October 2026.

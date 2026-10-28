"""Export a worker LoRA checkpoint (final.pt, a PEFT state dict) as a Hugging Face adapter folder.

Usage: python tools/export_hf.py CKPT OUT_DIR [BASE_MODEL] [LORA_R]
Runs on CPU. Writes adapter_config.json + adapter_model.safetensors, then reloads the folder
and checks every tensor round-trips and a sample prompt gives the same letter logits.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt, out = sys.argv[1], sys.argv[2]
base_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-1.5B-Instruct"
r = int(sys.argv[4]) if len(sys.argv) > 4 else 16

sd = torch.load(ckpt, map_location="cpu", weights_only=True)
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
m = get_peft_model(base, cfg)
res = m.load_state_dict(sd, strict=False)
lora_missing = [k for k in res.missing_keys if "lora_" in k]
print(f"loaded {len(sd)} tensors; unexpected={len(res.unexpected_keys)} lora_missing={len(lora_missing)}")
if res.unexpected_keys or lora_missing:
    sys.exit("checkpoint does not match the LoRA layout")
m.eval()

prompt = "You are playing Yu-Gi-Oh.\nOptions:\nA. Normal Summon Kashtira Fenrir\nB. Set a card\nC. End the turn\nAnswer:"
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
letters = [tok(prompt + " " + L, add_special_tokens=False).input_ids[-1] for L in "ABC"]
with torch.no_grad():
    before = m(ids).logits[0, -1, letters]

m.save_pretrained(out)
tok.save_pretrained(out)

base2 = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
m2 = PeftModel.from_pretrained(base2, out).eval()
sd2 = {k.replace(".default", ""): v for k, v in m2.state_dict().items() if "lora_" in k}
same = all(torch.equal(v, sd2[k.replace(".default", "")]) for k, v in sd.items())
with torch.no_grad():
    after = m2(ids).logits[0, -1, letters]
print(f"roundtrip tensors_equal={same} logits_before={before.tolist()} logits_after={after.tolist()} "
      f"max_diff={float((before - after).abs().max()):.2e}")
sys.exit(0 if same and torch.allclose(before, after, atol=1e-4) else 1)

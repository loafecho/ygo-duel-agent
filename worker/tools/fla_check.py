"""Score fixed prompts with the ZeroShot model; print choices, max logit gap and seconds. Run once without and once
with flash-linear-attention installed, then compare the two JSON files (same choices = safe to keep)."""
import json, sys, time
sys.argv = [sys.argv[0], "zeroshot"] + sys.argv[1:]
import worker  # noqa: E402  (app/worker.py)
prompts = json.load(open(sys.argv[2]))
sc = worker.Scorer(worker.ZEROSHOT_MODEL, chat=True, budget=2048)
sc.logits([p["prompt"] for p in prompts[:2]], [p["n"] for p in prompts[:2]])  # warm-up
worker.torch.cuda.synchronize()
t0 = time.time()
lg = sc.logits([p["prompt"] for p in prompts], [p["n"] for p in prompts])
worker.torch.cuda.synchronize()
secs = time.time() - t0
out = {"seconds": round(secs, 2), "choices": [int(l.argmax()) for l in lg], "logits": [l.tolist() for l in lg]}
json.dump(out, open(sys.argv[3], "w"))
print(json.dumps({"seconds": out["seconds"], "choices": out["choices"]}))

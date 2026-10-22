#!/usr/bin/python3 -I
"""ygo-worker supervisor: runs on the GPU workstation, one per mode, and owns the worker process.

Every CHECK_S seconds it runs the game guard. A game makes it send SIGTERM to the worker: the
worker writes a checkpoint and exits, so its CUDA context and VRAM are released. After
RESUME_CLEAN clean checks in a row it starts the worker again and the worker resumes.

The worker is a plain venv process (ROOT/venv/bin/python ROOT/app/worker.py <mode>), not a container.

The guard functions below are adapted from the guard of a co-tenant GPU job on the same host.
Changes: _ours() ("our" GPU processes are the worker and its children), and a
running co-tenant GPU job (YGO_YIELD_TO) counts as a game (its own guard would otherwise wait on our training forever).

A worker that exits 0 after writing <YGO_DATA>/done-<mode> is finished: the supervisor exits too.
Any other exit is a crash: restart after a pause, give up after 5 in a row.

Usage: supervisor.py train|zeroshot|table      Log: ~/.local/share/ygo-worker/logs/supervisor-<mode>.jsonl
Env: YGO_DATA (default ROOT/data), passed on to the worker with every other YGO_* variable.
"""
from __future__ import annotations

import json
import signal
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / ".local/share/ygo-worker"
CHECK_S = 2
RESUME_CLEAN = 2
GPU_BUSY = 50        # percent, same as the co-tenant GPU job: desktop apps run 20-25%
YIELD_TO = os.environ.get("YGO_YIELD_TO", "")  # script path suffix of a co-tenant GPU job that yields to games
MODE = sys.argv[1]
NAME = f"ygo-{MODE}"
DATA = Path(os.environ.get("YGO_DATA", str(ROOT / "data")))


def log(event: str, **kw) -> None:
    line = json.dumps({"t": round(time.time(), 3), "event": event, **kw})
    print(line, flush=True)
    with open(ROOT / "logs" / f"supervisor-{MODE}.jsonl", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------ guard (adapted)

def vram_used() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15)
    return int(r.stdout.strip().splitlines()[0])


def gpu_util() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15)
    return int(r.stdout.strip().splitlines()[0])


def _ours(pid: int) -> bool:
    """True when pid is the worker or one of its descendants (walks the parent chain)."""
    if _child is None:
        return False
    for _ in range(64):
        if pid == _child.pid:
            return True
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return False
        pid = int(stat.rsplit(")", 1)[1].split()[1])
        if pid <= 1:
            return False
    return False


def game_procs() -> list[str]:
    """Names of game-like processes: Steam's reaper with SteamLaunch, or gamescope wrapping a command."""
    hits = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if not argv:
            continue
        base = os.path.basename(argv[0])
        if base == "reaper" and "SteamLaunch" in argv:
            app = next((a for a in argv if a.startswith("AppId=")), "AppId=?")
            hits.append(f"steam:{app}")
        elif base == "gamescope" and "--" in argv:
            hits.append("gamescope")
        elif YIELD_TO and any(a.endswith(YIELD_TO) for a in argv[:3]):
            hits.append("gpu-job")
    return hits


def other_gpu_busy() -> int:
    """Highest SM% of any GPU process outside our worker (pmon), so our own load never counts."""
    r = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=20)
    top = 0
    for line in r.stdout.splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4 or not f[1].isdigit():
            continue
        if _ours(int(f[1])):
            continue
        sm = int(f[3]) if f[3].isdigit() else 0
        top = max(top, sm)
    return top


def game_reason(model_loaded: bool) -> str:
    g = game_procs()
    if g:
        return ",".join(sorted(set(g)))
    if model_loaded:
        busy = other_gpu_busy()
        return f"gpu-other:{busy}%" if busy > GPU_BUSY else ""
    util = gpu_util()
    return f"gpu:{util}%" if util > GPU_BUSY else ""


# ------------------------------------------------------------ worker process control

_child: subprocess.Popen | None = None


def running() -> bool:
    return _child is not None and _child.poll() is None


def start() -> None:
    global _child
    env = dict(os.environ,
               YGO_DATA=str(DATA),
               HF_HOME=str(ROOT / "hf"),
               YGO_BROKER=os.environ.get("YGO_BROKER", "http://198.51.100.10:8765"))
    out = open(ROOT / "logs" / f"worker-{MODE}.out", "a")
    _child = subprocess.Popen([str(ROOT / "venv/bin/python"), "-u", str(ROOT / "app/worker.py"), MODE] + sys.argv[2:],
                              env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    _child.t0 = time.time()
    log("started", pid=_child.pid)


def stop(reason: str) -> None:
    t0 = time.time()
    if running():
        _child.send_signal(signal.SIGTERM)
        try:
            _child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _child.kill()
            _child.wait()
    log("stopped", reason=reason, stop_s=round(time.time() - t0, 2), rc=_child.returncode, vram_mib=vram_used())


def on_term(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    log("supervisor_up", mode=MODE, baseline_vram_mib=vram_used(), data=str(DATA))
    clean = RESUME_CLEAN  # start straight away when no game is on
    paused_logged = False
    crashes = 0
    while True:
        if _child is not None and _child.poll() is not None and not getattr(_child, "seen_exit", False):
            _child.seen_exit = True
            rc = _child.returncode
            if rc == 0 and (DATA / f"done-{MODE}").exists():
                log("worker_done", rc=rc, result=(DATA / f"done-{MODE}").read_text().strip())
                return
            if rc != 0 and rc != -signal.SIGTERM:
                crashes += 1
                log("worker_crashed", rc=rc, crashes=crashes)
                if crashes >= int(os.environ.get("YGO_MAX_CRASHES", "5")):
                    log("giving_up", crashes=crashes)
                    return
                time.sleep(20)
        elif running():
            crashes = 0 if time.time() - getattr(_child, "t0", time.time()) > 600 else crashes
        up = running()
        reason = game_reason(model_loaded=up)
        if reason:
            clean = 0
            if up:
                log("game_detected", reason=reason)
                stop(reason)
            elif not paused_logged:
                log("waiting_for_game_end", reason=reason)
            paused_logged = True
        else:
            clean += 1
            if not up and clean >= RESUME_CLEAN:
                log("resume", after_clean_checks=clean)
                start()
                paused_logged = False
        time.sleep(CHECK_S)


if __name__ == "__main__":
    if MODE == "check":  # for pipeline.sh: "idle", or why the GPU is not ours to use (no worker running)
        print(game_reason(model_loaded=False) or "idle")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
    try:
        main()
    except KeyboardInterrupt:
        if running():
            stop("supervisor-exit")
        log("supervisor_down")

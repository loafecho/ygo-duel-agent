"""The supervisor's game guard (worker/supervisor.py) against real processes on this machine.

A fake game is a bash process started with argv[0] = "gamescope" (the guard reads /proc/<pid>/cmdline,
so this looks exactly like gamescope wrapping a command); a script whose path ends in YGO_YIELD_TO
stands in for a co-tenant GPU job.
"""
import os
import shutil
import signal
import subprocess
import time

from pytest import fixture

from conftest import load


@fixture
def supervisor(monkeypatch):
    monkeypatch.setenv("YGO_YIELD_TO", "fake-gpu-job.sh")
    return load("ygo_supervisor", "worker/supervisor.py", ["supervisor.py", "check"])


def _wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _fake(*argv):
    """bash running `sleep 30` with the given argv (argv[0] is what the guard sees as the program name)."""
    return subprocess.Popen(list(argv), executable=shutil.which("bash"), start_new_session=True)


def _end(p):
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()


def test_fake_gamescope_counts_as_a_game(supervisor):
    p = _fake("gamescope", "-c", "sleep 30; :", "fake-game", "--")
    try:
        assert _wait_for(lambda: "gamescope" in supervisor.game_procs())
    finally:
        _end(p)
    assert _wait_for(lambda: "gamescope" not in supervisor.game_procs())


def test_co_tenant_gpu_job_counts_as_a_game(supervisor, tmp_path):
    job = tmp_path / "fake-gpu-job.sh"
    job.write_text("sleep 30\n")
    p = subprocess.Popen(["bash", str(job)], start_new_session=True)
    try:
        assert _wait_for(lambda: "gpu-job" in supervisor.game_procs())
    finally:
        _end(p)


def test_gamescope_without_a_command_is_not_a_game(supervisor):
    p = _fake("gamescope", "-c", "sleep 30; :")
    try:
        time.sleep(0.3)
        assert "gamescope" not in supervisor.game_procs()
    finally:
        _end(p)


def test_no_worker_means_no_process_is_ours(supervisor):
    assert supervisor._child is None
    assert supervisor._ours(1) is False

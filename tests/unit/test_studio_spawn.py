"""Build subprocesses must start after the server has used PROJ (plan preview).

The generate page asks ``/api/plan`` before submitting a job, which loads PROJ and its
proj.db SQLite handle into the server process. A build started with fork() then ran PROJ's
at-fork handler in the child, which crashed with SIGSEGV on macOS before Python started
("build exited with code -11"). The check runs in a fresh interpreter: the test runner's
own process can mask the crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

SCRIPT = textwrap.dedent("""
    import os, sys, threading
    from worldgen.studio import plan_for, spawn_build

    # Like a request handler thread of the server.
    t = threading.Thread(target=lambda: plan_for({"center": {"lat": 25.033, "lon": 121.5654}, "radius_m": 1000}))
    t.start(); t.join()
    out = []

    def run():  # like the job runner thread
        p = spawn_build([sys.executable, "-c", "print('child ok')"], dict(os.environ))
        stdout, _ = p.communicate(timeout=60)
        out.append((stdout.strip(), p.returncode))

    t = threading.Thread(target=run)
    t.start(); t.join()
    print(out[0][0], out[0][1])
""")


def test_build_spawn_survives_proj_in_parent():
    path = os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH")) if p)
    env = dict(os.environ, PYTHONPATH=path)
    res = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().splitlines()[-1] == "child ok 0"

#!/usr/bin/env python
"""Run shell commands one after another, detached from the launching terminal.

    python scripts/job_queue.py --jobs results/queue/jobs_phase3.txt [--wait_for "run_transformers.py"]

Each non-empty, non-# line of the jobs file is `name | command`. Output of each job goes to
results/queue/<name>.log; a status line per job is appended to results/queue/status.txt
(START / DONE rc=<code> minutes). A failed job does not stop the queue. With --wait_for, the
queue first waits until no running process has that substring in its command line (one GPU job
at a time on the 4 GB card). Jobs already marked DONE rc=0 in status.txt are skipped on restart.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "results" / "queue"


def running(substr: str) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)\" }"],
                         capture_output=True, text=True).stdout
    mine = {str(os.getpid()), str(os.getppid())}
    for line in out.splitlines():
        parts = line.split("	", 2)
        if len(parts) == 3 and parts[0] not in mine and parts[1] not in mine and substr in parts[2]:
            return True
    return False


def done_jobs() -> set[str]:
    st = QDIR / "status.txt"
    if not st.exists():
        return set()
    done = set()
    for l in st.read_text(encoding="utf-8").splitlines():
        parts = l.split()
        if len(parts) >= 5 and parts[2] == "DONE" and parts[4] == "rc=0":
            done.add(parts[3])
    return done


def log_status(msg: str) -> None:
    with open(QDIR / "status.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--wait_for", default=None)
    args = ap.parse_args()
    QDIR.mkdir(parents=True, exist_ok=True)
    if args.wait_for:
        log_status(f"WAIT {args.wait_for}")
        while running(args.wait_for):
            time.sleep(60)
    env = dict(os.environ, PYTHONUTF8="1")
    skip = done_jobs()
    for line in Path(args.jobs).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, cmd = (x.strip() for x in line.partition("|"))
        if name in skip:
            log_status(f"SKIP {name} (already done)")
            continue
        log_status(f"START {name}")
        t0 = time.time()
        with open(QDIR / f"{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        log_status(f"DONE {name} rc={rc} {(time.time() - t0) / 60:.1f}min")
    log_status("QUEUE FINISHED")


if __name__ == "__main__":
    main()

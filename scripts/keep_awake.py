#!/usr/bin/env python
"""Keep Windows from idle-sleeping while long jobs run, without changing power settings.

    python scripts/keep_awake.py [--watch run_transformers.py job_queue.py train.py]

Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) - the same request media
players make - and re-checks every minute whether any watched process is still running. When none
is left (after a 10-minute grace period), it clears the request and exits, so the normal
sleep settings apply again. Closing the lid still sleeps the laptop (that is a separate setting).
Log: results/queue/keep_awake.log
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import time
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
LOG = Path(__file__).resolve().parents[1] / "results" / "queue" / "keep_awake.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def any_running(names: list[str]) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True).stdout
    return any(n in line and "keep_awake.py" not in line for line in out.splitlines() for n in names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="*", default=["run_transformers.py", "job_queue.py", "train.py"])
    ap.add_argument("--grace_minutes", type=float, default=10)
    args = ap.parse_args()
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log(f"START keeping system awake while any of {args.watch} runs")
    idle_since = None
    try:
        while True:
            if any_running(args.watch):
                idle_since = None
            else:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > args.grace_minutes * 60:
                    break
            k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(60)
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)
        log("STOP no watched job left; normal sleep settings apply again")


if __name__ == "__main__":
    main()

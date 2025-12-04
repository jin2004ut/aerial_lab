#!/usr/bin/env python3
"""
Batch-run evaluate.py over all .pt checkpoints in a run directory.

Usage:
    python scripts/run_batch_evaluate.py /full/path/to/logs/rsl_rl/beetle_hyper/2025-12-01_10-46-19
Optional args override defaults for evaluate.py (task, num_envs, device, video, video_length, headless).
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
from pathlib import Path


def extract_num(s: str):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def find_checkpoints(run_dir: Path):
    pts = [p for p in run_dir.glob("*.pt") if p.is_file()]

    # sort by numeric id in filename if present, otherwise lexicographic
    def key(p: Path):
        n = extract_num(p.stem)
        return (0, n) if n is not None else (1, p.name)

    return sorted(pts, key=key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, help="Run directory containing .pt files")
    p.add_argument("--task", type=str, default="Beetle-Pose-v0")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--video", action="store_true", default=False)
    p.add_argument("--video_length", type=int, default=500)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--evaluate_script", type=Path, default=Path("scripts/rsl_rl/evaluate.py"))
    p.add_argument("--extra", type=str, default="", help="Extra raw args string appended to command (optional).")
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"[ERROR] run_dir not found: {run_dir}")
        sys.exit(2)

    pts = find_checkpoints(run_dir)
    if not pts:
        print(f"[INFO] No .pt files found in {run_dir}")
        return

    if pts[0].stem == "model_0":
        print("[INFO] Detected 'model_0.pt' as first checkpoint, skipping it.")
        pts = pts[1:]

    print(f"[INFO] Found {len(pts)} checkpoints in {run_dir}")

    for ckpt in pts:
        cmd = [
            sys.executable,
            str(args.evaluate_script),
            f"--task={args.task}",
            f"--num_envs={args.num_envs}",
            f"--device={args.device}",
            f"--checkpoint={str(ckpt)}",
            "--kit_args=--/log/level=error --/log/quiet=1 --/exts/gpu.foundation.plugin/logging=0",
        ]
        if args.video:
            cmd.append("--video")
            cmd.append(f"--video_length={args.video_length}")
        if args.headless:
            cmd.append("--headless")
        if args.extra:
            # split simple extra string into tokens
            cmd.extend(args.extra.split())

        print("=" * 80)
        print(f"[INFO] Running evaluate for checkpoint: {ckpt.name}")
        print("COMMAND:", " ".join(map(str, cmd)))
        print("=" * 80)

        proc = subprocess.Popen(cmd)
        try:
            rc = proc.wait()
            print(f"[INFO] evaluate.py exited with return code: {rc}")
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt received: forwarding SIGINT to child and exiting.")
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            break

    print("[INFO] Batch evaluate finished or interrupted.")


if __name__ == "__main__":
    main()

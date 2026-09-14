#!/usr/bin/env python3
"""Prepare data, then run method/seed experiments with one worker per device.

Example:
    python scripts/run_main.py --devices cuda:0 cuda:1 --seeds 0 1 2 --resume

Relative config, data, and output paths are resolved from the repository root.
The active Python interpreter is used for every child command. Existing
checkpoints require --resume for training or --evaluate-only for evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import signal
import subprocess
import sys
from threading import Event, Lock

import yaml


class CommandRunner:
    """Track child process groups so cancellation also stops active training."""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.stopped = Event()
        self.lock = Lock()
        self.processes: set[subprocess.Popen] = set()

    def run(self, command: list[str], log_path: Path) -> int:
        with log_path.open("a", buffering=1) as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {shlex.join(command)}\n")
            with self.lock:
                if self.stopped.is_set():
                    return 130
                process = subprocess.Popen(
                    command, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                )
                self.processes.add(process)
            try:
                code = process.wait()
                log.write(f"[exit code {code}]\n")
                return code
            finally:
                # Keep interrupted preparation registered until cancel() can
                # stop its child; worker processes are registered likewise.
                if process.poll() is not None:
                    with self.lock:
                        self.processes.discard(process)

    def cancel(self) -> None:
        self.stopped.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
        with self.lock:
            self.processes.difference_update(processes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--kinds", nargs="+", choices=["drgw", "naive"], default=["drgw", "naive"])
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    args.devices = ["cuda:0" if device == "cuda" else device for device in args.devices]
    if any(not re.fullmatch(r"cpu|cuda:\d+", device) for device in args.devices):
        parser.error("--devices must contain cpu or CUDA device names such as cuda:0")
    args.devices = [device if device == "cpu" else f"cuda:{int(device.split(':')[1])}" for device in args.devices]
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    for name in ("devices", "seeds", "kinds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    return args


def run_worker(device, jobs, runner, args, config_path, output_dir, log_dir):
    results = []
    while not runner.stopped.is_set():
        try:
            kind, seed = jobs.get_nowait()
        except Empty:
            break
        log_path = log_dir / f"{kind}-seed-{seed}.log"
        checkpoint = output_dir / kind / f"seed-{seed}" / "checkpoint.pt"
        prefix = [sys.executable, "-u", "-m", "drgw"]
        common = ["--config", str(config_path), "--device", device]
        commands = []
        if not args.evaluate_only:
            train = prefix + ["train"] + common + ["--kind", kind, "--seed", str(seed)]
            if args.resume:
                train.append("--resume")
            commands.append(("train", train))
        if not args.train_only:
            commands.append(("evaluate", prefix + ["evaluate"] + common + ["--checkpoint", str(checkpoint)]))
        stage, code, error = "launch", 0, None
        try:
            for stage, command in commands:
                print(f"[{device}] {kind} seed={seed}: {stage} -> {log_path}", flush=True)
                code = runner.run(command, log_path)
                if code:
                    break
        except Exception as exc:
            code, error = 1, f"{type(exc).__name__}: {exc}"
        finally:
            jobs.task_done()
        result = dict(kind=kind, seed=seed, device=device, stage=stage, code=code, log=log_path, error=error)
        results.append(result)
        status = "complete" if code == 0 else f"FAILED ({stage}, exit {code})"
        print(f"[{device}] {kind} seed={seed}: {status}", flush=True)
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    try:
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        output_dir = Path(config["output_dir"]).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        log_dir = output_dir / "launcher"
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"Cannot load launcher configuration: {exc}", file=sys.stderr)
        return 1
    runner = CommandRunner(root)
    pool = None
    try:
        prepare_log = log_dir / "prepare.log"
        print(f"Preparing data -> {prepare_log}", flush=True)
        code = runner.run([sys.executable, "-u", "-m", "drgw", "prepare", "--config", str(config_path)], prepare_log)
        if code:
            print(f"Preparation FAILED (exit {code}); see {prepare_log}", file=sys.stderr)
            return 1
        jobs = Queue()
        for kind in args.kinds:
            for seed in args.seeds:
                jobs.put((kind, seed))
        pool = ThreadPoolExecutor(max_workers=len(args.devices))
        futures = [pool.submit(run_worker, device, jobs, runner, args, config_path, output_dir, log_dir)
                   for device in args.devices]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        failed = [result for result in results if result["code"] != 0]
        print(f"Finished: {len(results) - len(failed)} succeeded, {len(failed)} failed.", flush=True)
        for result in sorted(failed, key=lambda item: (item["kind"], item["seed"])):
            detail = f"; {result['error']}" if result["error"] else ""
            print(f"FAILED {result['kind']} seed={result['seed']} on {result['device']} "
                  f"during {result['stage']} (exit {result['code']}){detail}; log: {result['log']}", file=sys.stderr)
        return 1 if failed else 0
    except KeyboardInterrupt:
        runner.cancel()
        print("Interrupted; active experiment processes were stopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        runner.cancel()
        print(f"Launcher FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a fixed latency benchmark through Codex CLI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, TextIO


PROMPT = "不要调用工具。只输出 20 行连续编号，不要解释。"
WARMUP_RUNS = 2
MEASURED_RUNS = 10
TIMEOUT_SECONDS = 180
OUTPUT_DIR = Path(__file__).resolve().parent / "results-short"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_stdout(
    stream: TextIO, lines: queue.Queue[tuple[str | None, float]]
) -> None:
    try:
        for line in stream:
            lines.put((line, time.perf_counter()))
    finally:
        lines.put((None, time.perf_counter()))


def build_command(workdir: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-C",
        str(workdir),
        PROMPT,
    ]


def run_once(phase: str, run_number: int, workdir: Path, raw_dir: Path) -> dict[str, Any]:
    command = build_command(workdir)
    result: dict[str, Any] = {
        "engine": "codex",
        "phase": phase,
        "run": run_number,
        "first_event_ms": None,
        "waiting_time_ms": None,
        "total_ms": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "exit_code": None,
        "success": False,
        "timed_out": False,
        "error": None,
    }

    jsonl_path = raw_dir / f"{phase}_{run_number:03d}.jsonl"
    stderr_path = raw_dir / f"{phase}_{run_number:03d}.stderr.log"
    with jsonl_path.open("w", encoding="utf-8") as raw_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=workdir,
        )
        assert process.stdout is not None
        lines: queue.Queue[tuple[str | None, float]] = queue.Queue()
        reader = threading.Thread(target=read_stdout, args=(process.stdout, lines), daemon=True)
        reader.start()

        turn_completed = False
        while True:
            if time.perf_counter() - started > TIMEOUT_SECONDS:
                result["timed_out"] = True
                result["error"] = f"Timeout after {TIMEOUT_SECONDS}s"
                process.kill()
                break

            try:
                line, received_at = lines.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue

            if line is None:
                break

            raw_file.write(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            # Use the reader's timestamp instead of the later queue-consumption time.
            elapsed_ms = (received_at - started) * 1000
            if result["first_event_ms"] is None:
                result["first_event_ms"] = elapsed_ms

            event_type = event.get("type")
            item = event.get("item") or {}
            codex_message = (
                event_type == "item.completed" and item.get("type") == "agent_message"
            )
            if codex_message and result["waiting_time_ms"] is None:
                result["waiting_time_ms"] = elapsed_ms

            if event_type == "turn.completed":
                usage = event.get("usage") or {}
                result["input_tokens"] = usage.get("input_tokens")
                result["cached_input_tokens"] = usage.get("cached_input_tokens")
                result["output_tokens"] = usage.get("output_tokens")
                turn_completed = True

        result["exit_code"] = process.wait()

    result["total_ms"] = (time.perf_counter() - started) * 1000
    result["success"] = (
        result["exit_code"] == 0
        and not result["timed_out"]
        and turn_completed
        and result["waiting_time_ms"] is not None
    )
    if not result["success"] and result["error"] is None:
        result["error"] = "codex did not return a complete answer"
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    summary: dict[str, Any] = {
        "runs": len(rows),
        "successful_runs": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0,
    }
    waiting_times = [float(row["waiting_time_ms"]) for row in successful]
    summary["waiting_time_median_ms"] = statistics.median(waiting_times) if waiting_times else None
    summary["waiting_time_p90_ms"] = percentile(waiting_times, 0.9)

    for field in ("first_event_ms", "total_ms"):
        values = [float(row[field]) for row in successful if row[field] is not None]
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
        summary[f"{field}_p50"] = percentile(values, 0.5)
        summary[f"{field}_p90"] = percentile(values, 0.9)
    return summary


def display_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def display_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.2f} 秒"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Codex CLI with fixed settings."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    if shutil.which("codex") is None:
        print("Cannot find 'codex' in PATH.")
        return 2

    output_dir = OUTPUT_DIR / "codex"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="codex-bench-") as temporary_directory:
        workdir = Path(temporary_directory)
        for phase, count in (("warmup", WARMUP_RUNS), ("run", MEASURED_RUNS)):
            for run_number in range(1, count + 1):
                row = run_once(phase, run_number, workdir, raw_dir)
                all_rows.append(row)
                if phase == "run":
                    measured_rows.append(row)
                print(
                    f"[{phase} {run_number}/{count}] ok={row['success']} "
                    f"waiting={display_ms(row['waiting_time_ms'])}ms "
                    f"total={display_ms(row['total_ms'])}ms"
                )

    fields = list(all_rows[0])
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = summarize(measured_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"等待时间中位数：{display_seconds(summary['waiting_time_median_ms'])}\n"
        f"等待时间 P90：{display_seconds(summary['waiting_time_p90_ms'])}\n"
        f"成功率：{summary['success_rate']:.1%}"
    )
    print(f"Results: {output_dir}")
    return 0 if summary["success_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())

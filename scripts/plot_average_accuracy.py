#!/usr/bin/env python3
"""
Utility to visualize Average Accuracy curves from continual learning logs.

The script scans PEFT-CL log files, extracts the most recent run in each log
(identified by the last occurrence of "Learning on 0-10" by default), and plots
the Average Accuracy (CNN) reported after every incremental task.
"""

from __future__ import annotations

import argparse
import re
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Headless-friendly backend; still allows plt.show() if a display exists.
matplotlib = import_module("matplotlib")
matplotlib.use("Agg")
plt = import_module("matplotlib.pyplot")


AVERAGE_PATTERN = re.compile(r"Average Accuracy\s*\(CNN\):\s*([0-9.+-eE]+)")
TASK_RESULT_PATTERN = re.compile(
    r"Task\s+(\d+),\s*Epoch\s+\d+/\d+\s*=>.*?Train_accy\s+([0-9.+-eE]+)"
    r"(?:,\s*Test_accy\s+([0-9.+-eE]+))?",
    re.IGNORECASE,
)
CONFIG_TOKEN = "[trainer.py] => config:"

SpecialParser = Callable[[List[str], int, int], List[float]]
SPECIAL_METHOD_PARSERS: Dict[str, SpecialParser] = {}



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Average Accuracy curves for continual learning logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory that contains per-method log folders.",
    )
    parser.add_argument(
        "--dataset-subpath",
        type=str,
        default=None,
        help=(
            "Path (relative to each method folder) containing log files. "
            "If omitted, all *.log files in every subdirectory are scanned."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of method folder names to include. Defaults to every folder.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to read per run (x-axis length).",
    )
    parser.add_argument(
        "--start-token",
        type=str,
        default="Learning on 0-10",
        help="Marker that denotes the first task of a run. The last occurrence marks the run to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("average_accuracy_curves.png"),
        help="Output path for the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (useful when running locally with a display).",
    )
    return parser.parse_args()


def find_last_run_start(lines: List[str], start_token: str, method_name: str) -> int:
    """Return the line index of the last occurrence of the start token."""
    last_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if start_token in line:
            last_idx = idx
    if last_idx is not None:
        return last_idx

    if method_name in {"coscl", "seed"}:
        config_idx = find_last_config_idx(lines)
        if config_idx is not None:
            return config_idx

    return 0


def find_last_config_idx(lines: List[str]) -> Optional[int]:
    """Locate the line index of the last `[trainer.py] => config:` entry."""
    last_idx = None
    for idx, line in enumerate(lines):
        if CONFIG_TOKEN in line:
            last_idx = idx
    return last_idx


def extract_average_curve(
    lines: List[str], start_idx: int, max_tasks: int
) -> List[float]:
    """Extract Average Accuracy values starting from a given index."""
    values: List[float] = []
    for line in lines[start_idx:]:
        match = AVERAGE_PATTERN.search(line)
        if not match:
            continue
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
        if len(values) >= max_tasks:
            break
    return values


def extract_inflora_curve(lines: List[str], start_idx: int, max_tasks: int) -> List[float]:
    """Parse inflora logs that only report per-task train/test accuracy."""
    task_scores: Dict[int, float] = {}
    for line in lines[start_idx:]:
        match = TASK_RESULT_PATTERN.search(line)
        if not match:
            continue
        task_id = int(match.group(1))
        acc_str = match.group(3) or match.group(2)
        try:
            task_scores[task_id] = float(acc_str)
        except (TypeError, ValueError):
            continue

    if not task_scores:
        return []

    running_values: List[float] = []
    cumulative: List[float] = []
    for task_id in sorted(task_scores.keys()):
        cumulative.append(task_scores[task_id])
        running_values.append(sum(cumulative) / len(cumulative))
        if len(running_values) >= max_tasks:
            break

    return running_values


SPECIAL_METHOD_PARSERS["inflora"] = extract_inflora_curve


def parse_log_file(
    log_path: Path, method_name: str, start_token: str, max_tasks: int
) -> Optional[List[float]]:
    """Return Average Accuracy curve for the most recent run in a log file."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as err:
        print(f"[WARN] Failed to read {log_path}: {err}")
        return None

    start_idx = find_last_run_start(lines, start_token, method_name)
    parser = SPECIAL_METHOD_PARSERS.get(method_name, extract_average_curve)
    values = parser(lines, start_idx, max_tasks)
    return values if values else None


def iter_method_dirs(logs_dir: Path, methods_filter: Optional[Iterable[str]]) -> Iterable[Path]:
    for method_dir in sorted(logs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        if methods_filter and method_dir.name not in methods_filter:
            continue
        yield method_dir


def collect_curves(
    logs_dir: Path,
    dataset_subpath: Optional[Path],
    methods_filter: Optional[Iterable[str]],
    start_token: str,
    max_tasks: int,
) -> Dict[str, Tuple[List[float], Path]]:
    """Return {method_name: (curve, log_path)} for every method that has data."""
    results: Dict[str, Tuple[List[float], Path]] = {}
    for method_dir in iter_method_dirs(logs_dir, methods_filter):
        target_dir = method_dir / dataset_subpath if dataset_subpath else method_dir
        if not target_dir.exists():
            if dataset_subpath:
                print(f"[INFO] Skipping {method_dir.name}: missing {dataset_subpath}")
            else:
                print(f"[INFO] Skipping {method_dir.name}: directory missing")
            continue

        best_entry: Optional[Tuple[float, List[float], Path]] = None
        for log_path in sorted(target_dir.rglob("*.log")):
            values = parse_log_file(log_path, method_dir.name, start_token, max_tasks)
            if not values:
                continue
            mtime = log_path.stat().st_mtime
            if not best_entry or mtime > best_entry[0]:
                best_entry = (mtime, values, log_path)

        if best_entry:
            results[method_dir.name] = (best_entry[1], best_entry[2])
            print(
                f"[OK] {method_dir.name}: using {best_entry[2].relative_to(logs_dir)} "
                f"({len(best_entry[1])} tasks)"
            )
        else:
            print(f"[WARN] No usable runs found for {method_dir.name}")

    return results


def plot_curves(
    curves: Dict[str, Tuple[List[float], Path]],
    output_path: Path,
    max_tasks: int,
    show: bool,
) -> None:
    if not curves:
        raise SystemExit("No curves to plot. Did the log search return anything?")

    plt.figure(figsize=(10, 6))
    for method, (values, _) in sorted(curves.items()):
        x = list(range(1, len(values) + 1))
        plt.plot(x, values, marker="o", label=method)

    plt.xlabel("Task # (CIFAR100, 10 classes per task)")
    plt.ylabel("Average Accuracy (CNN, %)")
    # plt.title("Average Accuracy across Continual Learning Tasks")
    plt.xticks(range(1, max_tasks + 1))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left", bbox_to_anchor=(0, -0.15), ncol=3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Plot saved to {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_arguments()
    logs_dir = args.logs_dir.resolve()
    dataset_subpath = (
        Path(args.dataset_subpath.strip("/")) if args.dataset_subpath else None
    )
    methods_filter = set(args.methods) if args.methods else None

    print(f"[INFO] Searching logs under {logs_dir}")
    curves = collect_curves(
        logs_dir=logs_dir,
        dataset_subpath=dataset_subpath,
        methods_filter=methods_filter,
        start_token=args.start_token,
        max_tasks=args.max_tasks,
    )
    plot_curves(curves, args.output, args.max_tasks, args.show)


if __name__ == "__main__":
    main()


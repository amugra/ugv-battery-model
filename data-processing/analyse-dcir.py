#!/usr/bin/env python3
"""
Example Usage:
    python extract_r0.py current.csv voltage.csv -o results.csv

Optional tuning:
    --baseline-samples 20
    --pulse-start 20
    --stable-window 3
    --stable-tol 3
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


PULSE_RE = re.compile(r"^\s*#\s*Pulse\s+(\d+)\s*$", re.IGNORECASE)


@dataclass
class Sample:
    index: int
    hex_value: str
    decimal: int


@dataclass
class PulseBlock:
    pulse_number: int
    samples: List[Sample]


@dataclass
class PulseResult:
    pulse_number: int
    baseline_current_adc: float
    baseline_voltage_adc: float
    stable_triplet_start_offset: int
    stable_triplet_offsets: str
    stable_current_adc_avg: float
    stable_voltage_adc_avg: float
    dI_adc: float
    dV_adc: float
    baseline_current_a: float
    baseline_voltage_v: float
    stable_current_a: float
    stable_voltage_v: float
    dI_a: float
    dV_v: float
    r0_ohm: float
    r0_mohm: float


def adc_to_current(adc: float) -> float:
    return (((adc * 3.3) / 4096.0) - 0.02) / 0.02473


def adc_to_voltage(adc: float) -> float:
    return (((adc * 3.3) / 4096.0) - 0.02) / 0.5928


def parse_pulse_file(path: Path) -> List[PulseBlock]:
    blocks: List[PulseBlock] = []
    current_pulse_number: Optional[int] = None
    current_samples: List[Sample] = []
    expecting_column_header = False

    with path.open("r", newline="", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            pulse_match = PULSE_RE.match(line)
            if pulse_match:
                if current_pulse_number is not None:
                    blocks.append(PulseBlock(current_pulse_number, current_samples))
                current_pulse_number = int(pulse_match.group(1))
                current_samples = []
                expecting_column_header = True
                continue

            if expecting_column_header:
                # Skip the "index,hex,decimal" row
                expecting_column_header = False
                continue

            if current_pulse_number is None:
                raise ValueError(
                    f"{path}: encountered data before first pulse header at line {line_num}"
                )

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(
                    f"{path}: malformed data line {line_num}: expected 3 columns, got {len(parts)}"
                )

            try:
                idx = int(parts[0], 10)
                hex_value = parts[1]
                dec = int(parts[2], 10)
            except ValueError as e:
                raise ValueError(f"{path}: failed to parse line {line_num}: {line}") from e

            current_samples.append(Sample(index=idx, hex_value=hex_value, decimal=dec))

    if current_pulse_number is not None:
        blocks.append(PulseBlock(current_pulse_number, current_samples))

    return blocks


def find_first_quasi_stable_triplet(
    decimals: List[int],
    pulse_start: int,
    stable_window: int,
    stable_tol: int,
) -> Optional[int]:
    start = pulse_start
    end = len(decimals) - stable_window + 1

    for i in range(start, end):
        window = decimals[i:i + stable_window]
        if max(window) - min(window) <= stable_tol:
            return i - pulse_start

    return None


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot compute mean of empty list")
    return sum(vals) / len(vals)


def summarize(values: List[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "stdev": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def format_stats(name: str, values: List[float], unit: str = "") -> str:
    s = summarize(values)
    suffix = f" {unit}" if unit else ""
    if s["count"] == 0:
        return f"{name}: no valid values"

    return (
        f"{name}:\n"
        f"  count  = {s['count']}\n"
        f"  mean   = {s['mean']:.9f}{suffix}\n"
        f"  median = {s['median']:.9f}{suffix}\n"
        f"  stdev  = {s['stdev']:.9f}{suffix}\n"
        f"  min    = {s['min']:.9f}{suffix}\n"
        f"  max    = {s['max']:.9f}{suffix}"
    )


def process_pulses(
    current_blocks: List[PulseBlock],
    voltage_blocks: List[PulseBlock],
    baseline_samples: int,
    pulse_start: int,
    stable_window: int,
    stable_tol: int,
) -> List[PulseResult]:
    if len(current_blocks) != len(voltage_blocks):
        raise ValueError(
            f"Pulse count mismatch: current file has {len(current_blocks)} blocks, "
            f"voltage file has {len(voltage_blocks)} blocks"
        )

    results: List[PulseResult] = []

    for c_block, v_block in zip(current_blocks, voltage_blocks):
        if c_block.pulse_number != v_block.pulse_number:
            raise ValueError(
                f"Pulse numbering mismatch: current pulse {c_block.pulse_number}, "
                f"voltage pulse {v_block.pulse_number}"
            )

        if len(c_block.samples) != len(v_block.samples):
            raise ValueError(
                f"Sample count mismatch in pulse {c_block.pulse_number}: "
                f"current has {len(c_block.samples)}, voltage has {len(v_block.samples)}"
            )

        if len(c_block.samples) < pulse_start + stable_window:
            raise ValueError(
                f"Pulse {c_block.pulse_number} does not have enough samples "
                f"for pulse_start={pulse_start} and stable_window={stable_window}"
            )

        if baseline_samples > pulse_start:
            raise ValueError(
                f"baseline_samples ({baseline_samples}) must be <= pulse_start ({pulse_start})"
            )

        current_dec = [s.decimal for s in c_block.samples]
        voltage_dec = [s.decimal for s in v_block.samples]

        baseline_current_adc = mean(current_dec[:baseline_samples])
        baseline_voltage_adc = mean(voltage_dec[:baseline_samples])

        stable_offset = find_first_quasi_stable_triplet(
            current_dec,
            pulse_start=pulse_start,
            stable_window=stable_window,
            stable_tol=stable_tol,
        )

        if stable_offset is None:
            print(
                f"Warning: no quasi-stable triplet found in pulse {c_block.pulse_number}; skipping",
                file=sys.stderr,
            )
            continue

        stable_start = pulse_start + stable_offset
        stable_current_adc_avg = mean(current_dec[stable_start:stable_start + stable_window])
        stable_voltage_adc_avg = mean(voltage_dec[stable_start:stable_start + stable_window])

        dI_adc = stable_current_adc_avg - baseline_current_adc
        dV_adc = stable_voltage_adc_avg - baseline_voltage_adc

        baseline_current_a = adc_to_current(baseline_current_adc)
        stable_current_a = adc_to_current(stable_current_adc_avg)
        dI_a = stable_current_a - baseline_current_a

        baseline_voltage_v = adc_to_voltage(baseline_voltage_adc)
        stable_voltage_v = adc_to_voltage(stable_voltage_adc_avg)
        dV_v = stable_voltage_v - baseline_voltage_v

        if abs(dI_a) < 1e-15:
            r0_ohm = math.nan
            r0_mohm = math.nan
        else:
            # Negative sign makes a discharge step (voltage drop under positive current)
            # come out as positive resistance.
            r0_ohm = -(dV_v / dI_a)
            r0_mohm = r0_ohm * 1000.0

        stable_offsets = ",".join(str(stable_offset + i) for i in range(stable_window))

        results.append(
            PulseResult(
                pulse_number=c_block.pulse_number,
                baseline_current_adc=baseline_current_adc,
                baseline_voltage_adc=baseline_voltage_adc,
                stable_triplet_start_offset=stable_offset,
                stable_triplet_offsets=stable_offsets,
                stable_current_adc_avg=stable_current_adc_avg,
                stable_voltage_adc_avg=stable_voltage_adc_avg,
                dI_adc=dI_adc,
                dV_adc=dV_adc,
                baseline_current_a=baseline_current_a,
                baseline_voltage_v=baseline_voltage_v,
                stable_current_a=stable_current_a,
                stable_voltage_v=stable_voltage_v,
                dI_a=dI_a,
                dV_v=dV_v,
                r0_ohm=r0_ohm,
                r0_mohm=r0_mohm,
            )
        )

    return results


def write_results_csv(path: Path, results: List[PulseResult]) -> None:
    fieldnames = [
        "pulse_number",
        "baseline_current_adc",
        "baseline_voltage_adc",
        "stable_triplet_start_offset",
        "stable_triplet_offsets",
        "stable_current_adc_avg",
        "stable_voltage_adc_avg",
        "dI_adc",
        "dV_adc",
        "baseline_current_a",
        "baseline_voltage_v",
        "stable_current_a",
        "stable_voltage_v",
        "dI_a",
        "dV_v",
        "r0_ohm",
        "r0_mohm",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "pulse_number": r.pulse_number,
                "baseline_current_adc": f"{r.baseline_current_adc:.9f}",
                "baseline_voltage_adc": f"{r.baseline_voltage_adc:.9f}",
                "stable_triplet_start_offset": r.stable_triplet_start_offset,
                "stable_triplet_offsets": r.stable_triplet_offsets,
                "stable_current_adc_avg": f"{r.stable_current_adc_avg:.9f}",
                "stable_voltage_adc_avg": f"{r.stable_voltage_adc_avg:.9f}",
                "dI_adc": f"{r.dI_adc:.9f}",
                "dV_adc": f"{r.dV_adc:.9f}",
                "baseline_current_a": f"{r.baseline_current_a:.9f}",
                "baseline_voltage_v": f"{r.baseline_voltage_v:.9f}",
                "stable_current_a": f"{r.stable_current_a:.9f}",
                "stable_voltage_v": f"{r.stable_voltage_v:.9f}",
                "dI_a": f"{r.dI_a:.9f}",
                "dV_v": f"{r.dV_v:.9f}",
                "r0_ohm": f"{r.r0_ohm:.9f}" if not math.isnan(r.r0_ohm) else "nan",
                "r0_mohm": f"{r.r0_mohm:.9f}" if not math.isnan(r.r0_mohm) else "nan",
            })


def print_summary(results: List[PulseResult]) -> None:
    dI_adc = [r.dI_adc for r in results]
    dV_adc = [r.dV_adc for r in results]
    dI_a = [r.dI_a for r in results]
    dV_v = [r.dV_v for r in results]
    r0_ohm = [r.r0_ohm for r in results if not math.isnan(r.r0_ohm)]
    r0_mohm = [r.r0_mohm for r in results if not math.isnan(r.r0_mohm)]
    offsets = [float(r.stable_triplet_start_offset) for r in results]

    print("\n=== Per-dataset descriptive statistics ===")
    print(format_stats("Stable triplet start offset", offsets, "samples"))
    print()
    print(format_stats("dI (ADC ticks)", dI_adc))
    print()
    print(format_stats("dV (ADC ticks)", dV_adc))
    print()
    print(format_stats("dI (A)", dI_a, "A"))
    print()
    print(format_stats("dV (V)", dV_v, "V"))
    print()
    print(format_stats("R0 (ohm)", r0_ohm, "ohm"))
    print()
    print(format_stats("R0 (mohm)", r0_mohm, "mOhm"))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract R0 and descriptive statistics from paired current/voltage pulse files."
    )
    p.add_argument("current_file", type=Path, help="Path to current pulse dataset")
    p.add_argument("voltage_file", type=Path, help="Path to voltage pulse dataset")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("r0_results.csv"),
        help="Output CSV file for per-pulse results (default: r0_results.csv)",
    )
    p.add_argument(
        "--baseline-samples",
        type=int,
        default=20,
        help="Number of pre-pulse samples used for baseline averaging (default: 20)",
    )
    p.add_argument(
        "--pulse-start",
        type=int,
        default=20,
        help="Index offset inside each block where the pulse region begins (default: 20)",
    )
    p.add_argument(
        "--stable-window",
        type=int,
        default=3,
        help="Number of consecutive samples required for quasi-stable detection (default: 3)",
    )
    p.add_argument(
        "--stable-tol",
        type=int,
        default=3,
        help="Maximum spread in ADC ticks within the stable window (default: 3)",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    current_blocks = parse_pulse_file(args.current_file)
    voltage_blocks = parse_pulse_file(args.voltage_file)

    if not current_blocks:
        print("No pulse blocks found in current file.", file=sys.stderr)
        return 1
    if not voltage_blocks:
        print("No pulse blocks found in voltage file.", file=sys.stderr)
        return 1

    results = process_pulses(
        current_blocks=current_blocks,
        voltage_blocks=voltage_blocks,
        baseline_samples=args.baseline_samples,
        pulse_start=args.pulse_start,
        stable_window=args.stable_window,
        stable_tol=args.stable_tol,
    )

    if not results:
        print("No valid pulses processed.", file=sys.stderr)
        return 2

    write_results_csv(args.output, results)

    print(f"Processed {len(results)} pulses.")
    print(f"Per-pulse results written to: {args.output}")
    print_summary(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

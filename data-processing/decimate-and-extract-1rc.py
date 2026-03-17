#!/usr/bin/env python3
"""
Example:
    python3 extract_and_decimate_pulses.py input.csv output_dir

Optional:
    python3 extract_and_decimate_pulses.py input.csv output_dir \
        --value-col 2 \
        --baseline-center 2630 --baseline-tol 5 \
        --pulse-center 2570 --pulse-tol 5 \
        --sample-rate 800 \
        --pulse-seconds 60 \
        --baseline-confirm 20 \
        --pulse-confirm 8
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Row:
    raw: List[str]
    value: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract pulses from CSV and decimate 800 Hz -> 80 Hz."
    )
    p.add_argument("input_csv", help="Input CSV file")
    p.add_argument("output_dir", help="Directory for extracted pulse files")

    p.add_argument(
        "--value-col",
        type=int,
        default=2,
        help="0-based column index containing the decimal ADC value (default: 2)",
    )

    p.add_argument(
        "--baseline-center",
        type=float,
        default=2630.0,
        help="Baseline center value (default: 2630)",
    )
    p.add_argument(
        "--baseline-tol",
        type=float,
        default=5.0,
        help="Baseline tolerance ± value (default: 5)",
    )

    p.add_argument(
        "--pulse-center",
        type=float,
        default=2570.0,
        help="Pulse center value (default: 2570)",
    )
    p.add_argument(
        "--pulse-tol",
        type=float,
        default=5.0,
        help="Pulse tolerance ± value (default: 5)",
    )

    p.add_argument(
        "--sample-rate",
        type=float,
        default=800.0,
        help="Input sample rate in Hz (default: 800)",
    )
    p.add_argument(
        "--pulse-seconds",
        type=float,
        default=60.0,
        help="Pulse length in seconds (default: 60)",
    )

    p.add_argument(
        "--baseline-confirm",
        type=int,
        default=20,
        help=(
            "Require this many consecutive samples in baseline region before "
            "a pulse can start (default: 20)"
        ),
    )
    p.add_argument(
        "--pulse-confirm",
        type=int,
        default=8,
        help=(
            "Require this many consecutive samples in pulse region to confirm "
            "a pulse start (default: 8)"
        ),
    )

    p.add_argument(
        "--downsample-factor",
        type=int,
        default=10,
        help="Decimation factor for averaging (default: 10, so 800 Hz -> 80 Hz)",
    )

    p.add_argument(
        "--has-header",
        action="store_true",
        help="Set this if the CSV has a header row",
    )

    return p.parse_args()


def in_range(x: float, center: float, tol: float) -> bool:
    return (center - tol) <= x <= (center + tol)


def read_csv_rows(path: str, value_col: int, has_header: bool) -> Tuple[List[str], List[Row]]:
    header = []
    rows: List[Row] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)

        if has_header:
            try:
                header = next(reader)
            except StopIteration:
                return header, rows

        for line_num, r in enumerate(reader, start=2 if has_header else 1):
            if not r:
                continue
            if value_col >= len(r):
                raise ValueError(
                    f"Line {line_num}: value column {value_col} does not exist. "
                    f"Row has {len(r)} columns."
                )
            try:
                val = float(r[value_col].strip())
            except ValueError as e:
                raise ValueError(
                    f"Line {line_num}: could not parse value from column {value_col}: {r[value_col]!r}"
                ) from e

            rows.append(Row(raw=r, value=val))

    return header, rows


def find_pulse_starts(
    rows: List[Row],
    baseline_center: float,
    baseline_tol: float,
    pulse_center: float,
    pulse_tol: float,
    baseline_confirm: int,
    pulse_confirm: int,
    pulse_len_samples: int,
) -> List[int]:
    """
    Detect pulse starts using a simple state machine:

    - Wait until we have baseline_confirm consecutive samples in baseline region.
    - Then look for pulse_confirm consecutive samples in pulse region.
    - When found, mark the first sample of that pulse region as the pulse start.
    - Skip ahead by pulse_len_samples to avoid re-detecting inside the same pulse.
    """
    starts: List[int] = []
    n = len(rows)
    i = 0

    while i < n:
        # Need enough room for baseline confirm + pulse confirm
        if i + baseline_confirm + pulse_confirm >= n:
            break

        # Check baseline-confirm window ending at i + baseline_confirm - 1
        baseline_ok = True
        for j in range(i, i + baseline_confirm):
            if not in_range(rows[j].value, baseline_center, baseline_tol):
                baseline_ok = False
                break

        if not baseline_ok:
            i += 1
            continue

        # Search forward for transition into pulse region
        search_start = i + baseline_confirm
        found = False

        while search_start + pulse_confirm <= n:
            # Stop search if baseline no longer looks like "pre-pulse territory"
            # and we drift too far without finding pulse.
            # This keeps things from wandering forever through weird data.
            if search_start - i > pulse_len_samples:
                break

            pulse_ok = True
            for j in range(search_start, search_start + pulse_confirm):
                if not in_range(rows[j].value, pulse_center, pulse_tol):
                    pulse_ok = False
                    break

            if pulse_ok:
                starts.append(search_start)
                i = search_start + pulse_len_samples
                found = True
                break

            search_start += 1

        if not found:
            i += 1

    return starts


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_raw_pulse_csv(
    path: str,
    pulse_rows: List[Row],
    original_header: List[str],
    add_rel_time: bool,
    sample_rate: float,
):
    header = list(original_header) if original_header else []
    if add_rel_time:
        header = header + ["relative_sample", "relative_time_s"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        if header:
            writer.writerow(header)

        for k, row in enumerate(pulse_rows):
            out = list(row.raw)
            if add_rel_time:
                out += [k, f"{k / sample_rate:.9f}"]
            writer.writerow(out)


def write_decimated_pulse_csv(
    path: str,
    pulse_rows: List[Row],
    original_header: List[str],
    downsample_factor: int,
    input_sample_rate: float,
):
    """
    Average every N samples into one row.
    Keeps the first two original columns from the first row in the block if present,
    but replaces the decimal/value column with averaged value.
    Also appends block metadata.

    Since the user mainly cares about the processed curve, this is usually enough.
    """
    output_rate = input_sample_rate / downsample_factor

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        if original_header:
            out_header = list(original_header) + [
                "block_start_sample",
                "block_index",
                "relative_time_s",
                "averaged_value",
            ]
            writer.writerow(out_header)
        else:
            writer.writerow([
                "col0",
                "col1",
                "col2",
                "block_start_sample",
                "block_index",
                "relative_time_s",
                "averaged_value",
            ])

        block_index = 0
        for start in range(0, len(pulse_rows), downsample_factor):
            block = pulse_rows[start:start + downsample_factor]
            if len(block) < downsample_factor:
                # Drop incomplete last block for clean 10:1 decimation
                break

            avg_val = sum(r.value for r in block) / downsample_factor

            # Base output row: clone first raw row if possible
            base = list(block[0].raw)
            while len(base) < 3:
                base.append("")

            # Put averaged value into 3rd column position if it exists
            base[2] = f"{avg_val:.6f}"

            rel_time = block_index / output_rate
            out = base + [
                start,
                block_index,
                f"{rel_time:.9f}",
                f"{avg_val:.6f}",
            ]
            writer.writerow(out)
            block_index += 1


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    pulse_len_samples = int(round(args.sample_rate * args.pulse_seconds))

    if args.downsample_factor <= 0:
        raise ValueError("--downsample-factor must be > 0")

    if args.sample_rate % args.downsample_factor != 0:
        print(
            f"Warning: sample rate {args.sample_rate} is not an integer multiple of "
            f"downsample factor {args.downsample_factor}. Output rate will be "
            f"{args.sample_rate / args.downsample_factor:.6f} Hz."
        )

    header, rows = read_csv_rows(
        args.input_csv,
        value_col=args.value_col,
        has_header=args.has_header,
    )

    if not rows:
        print("No data rows found.")
        return

    starts = find_pulse_starts(
        rows=rows,
        baseline_center=args.baseline_center,
        baseline_tol=args.baseline_tol,
        pulse_center=args.pulse_center,
        pulse_tol=args.pulse_tol,
        baseline_confirm=args.baseline_confirm,
        pulse_confirm=args.pulse_confirm,
        pulse_len_samples=pulse_len_samples,
    )

    if not starts:
        print("No pulses found.")
        return

    print(f"Found {len(starts)} pulse(s).")
    print(f"Pulse length: {pulse_len_samples} samples ({args.pulse_seconds} s)")
    print(
        f"Downsampling: {args.sample_rate} Hz -> "
        f"{args.sample_rate / args.downsample_factor:.3f} Hz "
        f"(factor {args.downsample_factor})"
    )

    summary_path = os.path.join(args.output_dir, "pulse_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "pulse_number",
            "start_sample_in_input",
            "start_time_s_in_input",
            "raw_output_file",
            "decimated_output_file",
        ])

        for idx, start in enumerate(starts, start=1):
            end = start + pulse_len_samples
            if end > len(rows):
                print(
                    f"Pulse {idx}: skipped because it would run past EOF "
                    f"(start={start}, end={end}, len={len(rows)})"
                )
                continue

            pulse_rows = rows[start:end]

            raw_name = f"pulse_{idx:03d}_raw_800hz.csv"
            dec_name = f"pulse_{idx:03d}_decimated_80hz.csv"

            raw_path = os.path.join(args.output_dir, raw_name)
            dec_path = os.path.join(args.output_dir, dec_name)

            write_raw_pulse_csv(
                raw_path,
                pulse_rows,
                header,
                add_rel_time=True,
                sample_rate=args.sample_rate,
            )

            write_decimated_pulse_csv(
                dec_path,
                pulse_rows,
                header,
                downsample_factor=args.downsample_factor,
                input_sample_rate=args.sample_rate,
            )

            sw.writerow([
                idx,
                start,
                f"{start / args.sample_rate:.9f}",
                raw_name,
                dec_name,
            ])

            print(
                f"Pulse {idx:03d}: start sample {start}, "
                f"raw -> {raw_name}, decimated -> {dec_name}"
            )

    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

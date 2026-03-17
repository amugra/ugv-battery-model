#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import operator
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple


ConditionFunc = Callable[[int, int], bool]
ParsedRow = Tuple[int, str, int, List[str]]  # (index, hex_str, dec_val, original_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pulse trigger windows from a 3-column CSV data file."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument(
        "--baseline",
        type=int,
        default=2600,
        help="Baseline value required before pulse detection (default: 2600)",
    )
    parser.add_argument(
        "--condition",
        choices=["GT", "LT", "EQ"],
        default="LT",
        help="Comparison operator for pulse detection (default: LT)",
    )
    parser.add_argument(
        "--value",
        type=int,
        default=2400,
        help="Comparison value for pulse detection (default: 2400)",
    )
    parser.add_argument(
        "--pre",
        type=int,
        default=20,
        help="Number of samples before trigger to keep (default: 20)",
    )
    parser.add_argument(
        "--post",
        type=int,
        default=20,
        help="Number of samples after trigger to keep (default: 20)",
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=int,
        default=0,
        help=(
            "Allowed absolute deviation from baseline when arming detection. "
            "Example: baseline=2600 tolerance=20 accepts 2580..2620 (default: 0)"
        ),
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set this if the CSV has a header row",
    )
    return parser.parse_args()


def get_condition_func(name: str) -> ConditionFunc:
    return {
        "GT": operator.gt,
        "LT": operator.lt,
        "EQ": operator.eq,
    }[name]


def is_at_baseline(value: int, baseline: int, tolerance: int) -> bool:
    return abs(value - baseline) <= tolerance


def parse_row(row: List[str], row_num: int) -> Optional[ParsedRow]:
    if not row or all(cell.strip() == "" for cell in row):
        return None

    if len(row) < 3:
        raise ValueError(f"Row {row_num}: expected at least 3 columns, got {len(row)}")

    try:
        index = int(row[0].strip())
    except ValueError as exc:
        raise ValueError(f"Row {row_num}: invalid index '{row[0]}'") from exc

    hex_str = row[1].strip()

    try:
        dec_val = int(row[2].strip())
    except ValueError as exc:
        raise ValueError(f"Row {row_num}: invalid decimal value '{row[2]}'") from exc

    return index, hex_str, dec_val, row


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.is_file():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        return 1

    condition_func = get_condition_func(args.condition)

    pre_buffer: Deque[ParsedRow] = deque(maxlen=args.pre)
    armed = False
    collecting = False
    post_remaining = 0

    current_pulse: List[ParsedRow] = []
    pulse_count = 0
    total_written_rows = 0

    try:
        with input_path.open("r", encoding="utf-8", errors="replace", newline="") as infile, \
             output_path.open("w", encoding="utf-8", newline="") as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            if args.has_header:
                next(reader, None)

            for row_num, row in enumerate(reader, start=2 if args.has_header else 1):
                parsed = parse_row(row, row_num)
                if parsed is None:
                    continue

                index, hex_str, dec_val, original_row = parsed

                if collecting:
                    current_pulse.append(parsed)

                    if post_remaining > 0:
                        post_remaining -= 1

                    if post_remaining == 0:
                        pulse_count += 1
                        writer.writerow([f"# Pulse {pulse_count}"])
                        writer.writerow(["index", "hex_value", "decimal_value"])

                        for pulse_row in current_pulse:
                            writer.writerow(pulse_row[3])
                            total_written_rows += 1

                        writer.writerow([])
                        current_pulse.clear()
                        collecting = False
                        armed = False
                        pre_buffer.clear()

                    continue

                if is_at_baseline(dec_val, args.baseline, args.baseline_tolerance):
                    armed = True

                if armed and condition_func(dec_val, args.value):
                    current_pulse = list(pre_buffer)
                    current_pulse.append(parsed)
                    collecting = True
                    post_remaining = args.post
                    continue

                pre_buffer.append(parsed)

    except ValueError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Pulses found: {pulse_count}")
    print(f"Rows written: {total_written_rows}")
    print(f"Output file: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



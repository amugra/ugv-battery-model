#!/usr/bin/env python3
"""
This script analyzes the OFF / RELAXATION part of the cycle and builds
a relaxation-derived resistance table using:

    R_relax(t) = (V(t) - V_loaded) / I_loaded

where:
- V_loaded is the average voltage in a stable region just before turn-off
- I_loaded is the average current in a stable region just before turn-off
- V(t) is the voltage at time t after turn-off

Example usage:
    python pulse_relaxation_dcr.py current.csv voltage.csv results.csv

Example:
    python pulse_relaxation_dcr.py current.csv voltage.csv results.csv \
        --sample-rate 800 \
        --relax-times 0.001,0.01,0.1,1,10
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# ----------------------------
# Transfer functions
# ----------------------------

def adc_to_current_amps(adc_reading: float) -> float:
    """Convert current ADC ticks to amps."""
    return (((adc_reading * 3.3) / 4096.0) - 0.02) / 0.02473


def adc_to_voltage_volts(adc_reading: float) -> float:
    """Convert voltage ADC ticks to volts."""
    return (((adc_reading * 3.3) / 4096.0) - 0.02) / 0.5928


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Pulse:
    number: int
    start_idx: int
    end_idx: int


# ----------------------------
# CSV loading
# ----------------------------

def load_adc_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV with columns: index, hex, decimal
    Returns DataFrame with:
        sample_index, hex, adc
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}") from e

    if df.shape[1] < 3:
        raise ValueError(
            f"File '{path}' must have at least 3 columns: index, hex, decimal"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["sample_index", "hex", "adc"]

    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df["adc"] = pd.to_numeric(df["adc"], errors="coerce")

    df = df.dropna(subset=["sample_index", "adc"]).reset_index(drop=True)
    df["sample_index"] = df["sample_index"].astype(int)

    return df


# ----------------------------
# Pulse detection
# ----------------------------

def find_pulses(
    current_adc: pd.Series,
    on_threshold: float,
    off_threshold: float,
    min_width_samples: int,
) -> List[Pulse]:
    """
    Detect pulses from current ADC:
    - pulse starts when signal crosses from below on_threshold to >= on_threshold
    - pulse ends when signal falls below off_threshold
    """
    pulses: List[Pulse] = []
    values = current_adc.to_numpy()

    in_pulse = False
    start_idx: Optional[int] = None
    pulse_number = 0

    for i in range(1, len(values)):
        prev_val = values[i - 1]
        curr_val = values[i]

        if not in_pulse:
            if prev_val < on_threshold <= curr_val:
                in_pulse = True
                start_idx = i
        else:
            if curr_val < off_threshold:
                end_idx = i - 1
                if start_idx is not None and (end_idx - start_idx + 1) >= min_width_samples:
                    pulse_number += 1
                    pulses.append(Pulse(pulse_number, start_idx, end_idx))
                in_pulse = False
                start_idx = None

    if in_pulse and start_idx is not None:
        end_idx = len(values) - 1
        if (end_idx - start_idx + 1) >= min_width_samples:
            pulse_number += 1
            pulses.append(Pulse(pulse_number, start_idx, end_idx))

    return pulses


# ----------------------------
# Helpers
# ----------------------------

def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return float("nan")
    return float(series.std())


def sample_offset_for_time(t_seconds: float, sample_rate_hz: float) -> int:
    """
    Convert target time to nearest sample offset.
    At 800 Hz:
        0.001 s -> round(0.8) = 1 sample
    """
    return int(round(t_seconds * sample_rate_hz))


def floor_time_for_reporting(duration_s: float, step_s: float = 0.25) -> float:
    """
    Round duration down to a reporting step.
    Example:
        9.86625 -> 9.75 when step_s=0.25
    """
    if duration_s <= 0:
        return 0.0
    return math.floor(duration_s / step_s) * step_s


def compute_stable_region_from_end(
    df: pd.DataFrame,
    stable_window: int,
    current_std_threshold_a: float,
    voltage_std_threshold_v: float,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Search backwards through a region for a stable window.
    Returns (start_local_idx, end_local_idx), region-local indexing.
    """
    n = len(df)
    if n < stable_window:
        return None, None

    for start in range(n - stable_window, -1, -1):
        end = start + stable_window
        window = df.iloc[start:end]

        i_std = safe_std(window["current_a"])
        v_std = safe_std(window["voltage_v"])

        if (
            math.isfinite(i_std)
            and math.isfinite(v_std)
            and i_std <= current_std_threshold_a
            and v_std <= voltage_std_threshold_v
        ):
            return start, end - 1

    return None, None


def compute_stable_region_from_start(
    df: pd.DataFrame,
    stable_window: int,
    current_std_threshold_a: float,
    voltage_std_threshold_v: float,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Search forwards through a region for a stable window.
    Returns (start_local_idx, end_local_idx), region-local indexing.
    Useful for finding a stable late-off / recovery region.
    """
    n = len(df)
    if n < stable_window:
        return None, None

    for start in range(0, n - stable_window + 1):
        end = start + stable_window
        window = df.iloc[start:end]

        i_std = safe_std(window["current_a"])
        v_std = safe_std(window["voltage_v"])

        if (
            math.isfinite(i_std)
            and math.isfinite(v_std)
            and i_std <= current_std_threshold_a
            and v_std <= voltage_std_threshold_v
        ):
            return start, end - 1

    return None, None


# ----------------------------
# Analysis
# ----------------------------

def analyze_relaxation(
    current_df: pd.DataFrame,
    voltage_df: pd.DataFrame,
    sample_rate_hz: float,
    current_on_threshold: float,
    current_off_threshold: float,
    min_width_samples: int,
    stable_window: int,
    current_stable_std_a: float,
    voltage_stable_std_v: float,
    relax_times: List[float],
    duration_round_step_s: float,
    min_off_samples: int,
) -> pd.DataFrame:
    if len(current_df) != len(voltage_df):
        raise ValueError(
            f"Current and voltage files have different lengths: "
            f"{len(current_df)} vs {len(voltage_df)}"
        )

    merged = pd.DataFrame({
        "sample_index": current_df["sample_index"],
        "current_adc": current_df["adc"],
        "voltage_adc": voltage_df["adc"],
    })

    merged["current_a"] = merged["current_adc"].apply(adc_to_current_amps)
    merged["voltage_v"] = merged["voltage_adc"].apply(adc_to_voltage_volts)
    merged["time_s"] = merged.index / sample_rate_hz

    pulses = find_pulses(
        merged["current_adc"],
        on_threshold=current_on_threshold,
        off_threshold=current_off_threshold,
        min_width_samples=min_width_samples,
    )

    if len(pulses) < 1:
        raise RuntimeError("No pulses detected. Check thresholds or input data.")

    # Build off regions: from pulse end+1 to next pulse start-1, or EOF for last pulse
    off_regions = []
    for i, pulse in enumerate(pulses):
        off_start = pulse.end_idx + 1
        if i < len(pulses) - 1:
            off_end = pulses[i + 1].start_idx - 1
        else:
            off_end = len(merged) - 1

        if off_end >= off_start:
            off_len = off_end - off_start + 1
            if off_len >= min_off_samples:
                off_regions.append((pulse, off_start, off_end))

    if not off_regions:
        raise RuntimeError("No valid off/relaxation regions found after pulses.")

    off_durations_s = [
        (off_end - off_start + 1) / sample_rate_hz
        for _, off_start, off_end in off_regions
    ]
    avg_off_duration_s = sum(off_durations_s) / len(off_durations_s)
    effective_off_time_s = floor_time_for_reporting(
        avg_off_duration_s,
        step_s=duration_round_step_s,
    )

    if effective_off_time_s <= 0:
        raise RuntimeError("Computed effective off-time is <= 0. Check detection settings.")

    results = []

    for pulse, off_start, off_end in off_regions:
        pulse_df = merged.iloc[pulse.start_idx:pulse.end_idx + 1].copy()
        off_df = merged.iloc[off_start:off_end + 1].copy()

        pulse_duration_s = (pulse.end_idx - pulse.start_idx + 1) / sample_rate_hz
        off_duration_s = (off_end - off_start + 1) / sample_rate_hz

        # Find stable loaded region just before turn-off
        loaded_stable_start_local, loaded_stable_end_local = compute_stable_region_from_end(
            pulse_df,
            stable_window=stable_window,
            current_std_threshold_a=current_stable_std_a,
            voltage_std_threshold_v=voltage_stable_std_v,
        )

        if loaded_stable_start_local is not None and loaded_stable_end_local is not None:
            loaded_stable_df = pulse_df.iloc[loaded_stable_start_local:loaded_stable_end_local + 1]
        else:
            # fallback: use last stable_window samples of pulse if available
            loaded_stable_df = pulse_df.iloc[max(0, len(pulse_df) - stable_window):]

        if len(loaded_stable_df) == 0:
            continue

        v_loaded = safe_mean(loaded_stable_df["voltage_v"])
        i_loaded = safe_mean(loaded_stable_df["current_a"])

        # Find a stable late off / recovery region
        off_stable_start_local, off_stable_end_local = compute_stable_region_from_end(
            off_df,
            stable_window=stable_window,
            current_std_threshold_a=current_stable_std_a,
            voltage_std_threshold_v=voltage_stable_std_v,
        )

        if off_stable_start_local is not None and off_stable_end_local is not None:
            off_stable_df = off_df.iloc[off_stable_start_local:off_stable_end_local + 1]
            v_recovered = safe_mean(off_stable_df["voltage_v"])
            i_recovered = safe_mean(off_stable_df["current_a"])
            r_relax_total = (v_recovered - v_loaded) / i_loaded if abs(i_loaded) > 1e-12 else float("nan")
        else:
            v_recovered = float("nan")
            i_recovered = float("nan")
            r_relax_total = float("nan")

        row = {
            "pulse_number": pulse.number,
            "pulse_start_sample": pulse.start_idx,
            "pulse_end_sample": pulse.end_idx,
            "off_start_sample": off_start,
            "off_end_sample": off_end,
            "pulse_start_time_s": pulse.start_idx / sample_rate_hz,
            "pulse_end_time_s": pulse.end_idx / sample_rate_hz,
            "off_start_time_s": off_start / sample_rate_hz,
            "off_end_time_s": off_end / sample_rate_hz,
            "pulse_duration_s": pulse_duration_s,
            "off_duration_s": off_duration_s,
            "avg_off_duration_s": avg_off_duration_s,
            "effective_off_time_s": effective_off_time_s,
            "loaded_stable_samples": len(loaded_stable_df),
            "v_loaded_v": v_loaded,
            "i_loaded_a": i_loaded,
            "v_drop_loaded_to_recovered_v": (v_recovered - v_loaded) if math.isfinite(v_recovered) else float("nan"),
        }

        if loaded_stable_start_local is not None and loaded_stable_end_local is not None:
            row["loaded_stable_start_sample"] = pulse.start_idx + loaded_stable_start_local
            row["loaded_stable_end_sample"] = pulse.start_idx + loaded_stable_end_local
        else:
            row["loaded_stable_start_sample"] = pulse.end_idx - len(loaded_stable_df) + 1
            row["loaded_stable_end_sample"] = pulse.end_idx

        # Relaxation resistance table
        for t in relax_times:
            col_r = f"R_relax_{t:g}s_ohm"
            col_v = f"V_relax_{t:g}s_v"
            col_i = f"I_relax_{t:g}s_a"
            col_sample = f"relax_sample_offset_{t:g}s"
            col_eval_t = f"actual_relax_eval_time_{t:g}s"
            col_dv = f"dV_relax_{t:g}s_v"

            eval_time_s = t
            if t > off_duration_s:
                eval_time_s = effective_off_time_s

            sample_offset = sample_offset_for_time(eval_time_s, sample_rate_hz)

            if len(off_df) == 0:
                row[col_sample] = float("nan")
                row[col_eval_t] = float("nan")
                row[col_v] = float("nan")
                row[col_i] = float("nan")
                row[col_dv] = float("nan")
                row[col_r] = float("nan")
                continue

            if sample_offset >= len(off_df):
                sample_offset = len(off_df) - 1
            if sample_offset < 0:
                sample_offset = 0

            point = off_df.iloc[sample_offset]
            v_t = float(point["voltage_v"])
            i_t = float(point["current_a"])
            dv_recovered = v_t - v_loaded

            if abs(i_loaded) > 1e-12:
                r_t = dv_recovered / i_loaded
            else:
                r_t = float("nan")

            row[col_sample] = sample_offset
            row[col_eval_t] = eval_time_s
            row[col_v] = v_t
            row[col_i] = i_t
            row[col_dv] = dv_recovered
            row[col_r] = r_t

        # Late stable off region summary
        if off_stable_start_local is not None and off_stable_end_local is not None:
            row["off_stable_start_sample"] = off_start + off_stable_start_local
            row["off_stable_end_sample"] = off_start + off_stable_end_local
            row["off_stable_samples"] = len(off_stable_df)
            row["v_recovered_v"] = v_recovered
            row["i_recovered_a"] = i_recovered
            row["R_relax_total_ohm"] = r_relax_total
        else:
            row["off_stable_start_sample"] = float("nan")
            row["off_stable_end_sample"] = float("nan")
            row["off_stable_samples"] = 0
            row["v_recovered_v"] = float("nan")
            row["i_recovered_a"] = float("nan")
            row["R_relax_total_ohm"] = float("nan")

        results.append(row)

    return pd.DataFrame(results)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer a relaxation-derived resistance table from the OFF portion after each current pulse."
    )

    parser.add_argument("current_csv", type=Path, help="Current pulse CSV file")
    parser.add_argument("voltage_csv", type=Path, help="Voltage pulse CSV file")
    parser.add_argument("output_csv", type=Path, help="Output results CSV file")

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=800.0,
        help="Sampling rate in Hz (default: 800)"
    )
    parser.add_argument(
        "--current-on-threshold",
        type=float,
        default=600.0,
        help="Current ADC threshold for pulse start (default: 600)"
    )
    parser.add_argument(
        "--current-off-threshold",
        type=float,
        default=100.0,
        help="Current ADC threshold for pulse end (default: 100)"
    )
    parser.add_argument(
        "--min-width-samples",
        type=int,
        default=8,
        help="Minimum pulse width in samples to keep (default: 8)"
    )
    parser.add_argument(
        "--min-off-samples",
        type=int,
        default=8,
        help="Minimum OFF region length in samples to keep (default: 8)"
    )
    parser.add_argument(
        "--stable-window",
        type=int,
        default=40,
        help="Window length used to find stable loaded/off regions (default: 40)"
    )
    parser.add_argument(
        "--current-stable-std-a",
        type=float,
        default=0.05,
        help="Max current std-dev for stable region in amps (default: 0.05)"
    )
    parser.add_argument(
        "--voltage-stable-std-v",
        type=float,
        default=0.005,
        help="Max voltage std-dev for stable region in volts (default: 0.005)"
    )
    parser.add_argument(
        "--relax-times",
        type=str,
        default="0.001,0.01,0.1,1,10,30,60",
        help="Comma-separated relaxation evaluation times in seconds"
    )
    parser.add_argument(
        "--duration-round-step",
        type=float,
        default=0.25,
        help="Step for rounding down average off-time, in seconds (default: 0.25)"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        relax_times = [float(x.strip()) for x in args.relax_times.split(",") if x.strip()]
        relax_times = sorted(set(relax_times))
    except ValueError:
        print("Error: invalid --relax-times value", file=sys.stderr)
        return 1

    try:
        current_df = load_adc_csv(args.current_csv)
        voltage_df = load_adc_csv(args.voltage_csv)

        results_df = analyze_relaxation(
            current_df=current_df,
            voltage_df=voltage_df,
            sample_rate_hz=args.sample_rate,
            current_on_threshold=args.current_on_threshold,
            current_off_threshold=args.current_off_threshold,
            min_width_samples=args.min_width_samples,
            stable_window=args.stable_window,
            current_stable_std_a=args.current_stable_std_a,
            voltage_stable_std_v=args.voltage_stable_std_v,
            relax_times=relax_times,
            duration_round_step_s=args.duration_round_step,
            min_off_samples=args.min_off_samples,
        )

        results_df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

        print(f"Processed {len(results_df)} relaxation segment(s).")
        print(f"Average off duration: {results_df['avg_off_duration_s'].iloc[0]:.6f} s")
        print(f"Effective off time:   {results_df['effective_off_time_s'].iloc[0]:.6f} s")
        print(f"Results written to: {args.output_csv}")

        if len(results_df) > 0:
            print("\nPreview:")
            print(results_df.head().to_string(index=False))

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from battery_model import write_mock_measurements_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock instrumentation source for the UGV battery EKF.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--duration", type=int, default=900, help="Duration in seconds.")
    parser.add_argument(
        "--cycle",
        choices=["transfer", "precision", "heavy"],
        default="heavy",
        help="Cycle type to simulate.",
    )
    parser.add_argument("--current-noise", type=float, default=0.15)
    parser.add_argument("--voltage-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = write_mock_measurements_csv(
        output_csv=Path(args.output),
        duration_s=args.duration,
        cycle_name=args.cycle,
        current_noise_std_a=args.current_noise,
        voltage_noise_std_v=args.voltage_noise,
        seed=args.seed,
    )
    print(f"Mock measurements written to {path}")

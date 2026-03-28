from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from battery_model import write_mock_measurements_csv
from ekf_estimator import run_ekf_from_csv


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def merge_rows(meas_rows: List[Dict[str, str]], ekf_rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    ekf_by_time = {int(float(r['time_s'])): r for r in ekf_rows}
    merged: List[Dict[str, float]] = []
    for mr in meas_rows:
        t = int(float(mr['time_s']))
        er = ekf_by_time.get(t)
        if er is None:
            continue
        merged.append(
            {
                'time_s': t,
                'current_true_a': float(mr['current_true_a']),
                'current_meas_a': float(mr['current_meas_a']),
                'terminal_true_v': float(mr['terminal_true_v']),
                'terminal_meas_v': float(mr['terminal_meas_v']),
                'soc_true_percent': float(mr['soc_true_percent']),
                'power_true_w': float(mr['power_true_w']),
                'soc_est_percent': float(er['soc_est_percent']),
                'terminal_est_v': float(er['terminal_est_v']),
                'power_est_w': float(er['power_est_w']),
                'innovation_v': float(er['innovation_v']),
                'var_soc': float(er['var_soc']),
                'var_v1': float(er['var_v1']),
            }
        )
    return merged


def write_summary(path: Path, merged: List[Dict[str, float]], cycle_name: str) -> Path:
    if not merged:
        raise ValueError('No merged rows available for summary.')

    n = len(merged)
    soc_err_abs = [abs(r['soc_est_percent'] - r['soc_true_percent']) for r in merged]
    volt_err_abs = [abs(r['terminal_est_v'] - r['terminal_true_v']) for r in merged]
    avg_power_true = sum(r['power_true_w'] for r in merged) / n
    avg_power_est = sum(r['power_est_w'] for r in merged) / n
    final = merged[-1]

    with path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'cycle_name',
                'samples',
                'duration_s',
                'avg_true_power_w',
                'avg_est_power_w',
                'final_soc_true_percent',
                'final_soc_est_percent',
                'mean_abs_soc_error_percent',
                'max_abs_soc_error_percent',
                'mean_abs_voltage_error_v',
                'max_abs_voltage_error_v',
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                'cycle_name': cycle_name,
                'samples': n,
                'duration_s': merged[-1]['time_s'],
                'avg_true_power_w': round(avg_power_true, 6),
                'avg_est_power_w': round(avg_power_est, 6),
                'final_soc_true_percent': round(final['soc_true_percent'], 6),
                'final_soc_est_percent': round(final['soc_est_percent'], 6),
                'mean_abs_soc_error_percent': round(sum(soc_err_abs) / n, 6),
                'max_abs_soc_error_percent': round(max(soc_err_abs), 6),
                'mean_abs_voltage_error_v': round(sum(volt_err_abs) / n, 6),
                'max_abs_voltage_error_v': round(max(volt_err_abs), 6),
            }
        )
    return path


def make_plot(path: Path, merged: List[Dict[str, float]], cycle_name: str) -> Path:
    times = [r['time_s'] for r in merged]

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(times, [r['current_true_a'] for r in merged], label='Current true [A]')
    plt.plot(times, [r['current_meas_a'] for r in merged], label='Current measured [A]', alpha=0.7)
    plt.ylabel('Current [A]')
    plt.title(f'UGV battery EKF demo - {cycle_name} cycle')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(times, [r['terminal_true_v'] for r in merged], label='Terminal true [V]')
    plt.plot(times, [r['terminal_meas_v'] for r in merged], label='Terminal measured [V]', alpha=0.7)
    plt.plot(times, [r['terminal_est_v'] for r in merged], label='Terminal estimated [V]', alpha=0.9)
    plt.ylabel('Voltage [V]')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(times, [r['soc_true_percent'] for r in merged], label='SOC true [%]')
    plt.plot(times, [r['soc_est_percent'] for r in merged], label='SOC estimated [%]', alpha=0.9)
    plt.ylabel('SOC [%]')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(times, [r['power_true_w'] for r in merged], label='Power true [W]')
    plt.plot(times, [r['power_est_w'] for r in merged], label='Power estimated [W]', alpha=0.9)
    plt.ylabel('Power [W]')
    plt.xlabel('Time [s]')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='End-to-end EKF demo: generate mock measurements, run EKF, and create plots/summary.'
    )
    parser.add_argument('--cycle', choices=['transfer', 'precision', 'heavy'], default='heavy')
    parser.add_argument('--duration', type=int, default=900, help='Simulation duration in seconds.')
    parser.add_argument('--dt', type=float, default=1.0, help='Estimator timestep in seconds.')
    parser.add_argument('--current-noise', type=float, default=0.15)
    parser.add_argument('--voltage-noise', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='.', help='Directory for generated files.')
    parser.add_argument('--prefix', default='demo', help='Filename prefix.')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    meas_path = outdir / f'{args.prefix}_{args.cycle}_measurements.csv'
    ekf_path = outdir / f'{args.prefix}_{args.cycle}_ekf.csv'
    plot_path = outdir / f'{args.prefix}_{args.cycle}_plot.png'
    summary_path = outdir / f'{args.prefix}_{args.cycle}_summary.csv'

    write_mock_measurements_csv(
        output_csv=meas_path,
        duration_s=args.duration,
        cycle_name=args.cycle,
        current_noise_std_a=args.current_noise,
        voltage_noise_std_v=args.voltage_noise,
        seed=args.seed,
    )
    run_ekf_from_csv(meas_path, ekf_path, dt_s=args.dt)

    meas_rows = read_csv_rows(meas_path)
    ekf_rows = read_csv_rows(ekf_path)
    merged = merge_rows(meas_rows, ekf_rows)

    make_plot(plot_path, merged, args.cycle)
    write_summary(summary_path, merged, args.cycle)

    print(f'Measurements: {meas_path}')
    print(f'EKF output:    {ekf_path}')
    print(f'Plot:          {plot_path}')
    print(f'Summary:       {summary_path}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class BatteryParams:
    cells_series: int = 14
    capacity_ah: float = 94.0
    nominal_pack_voltage_v: float = 48.0
    initial_cell_voltage_v: float = 4.0
    initial_pack_voltage_v: float = 56.0

    # Cell-level ECM parameters from the user.
    dcir_cell_ohm: float = 4.6e-3
    r0_cell_ohm: float = 2.76e-3
    r1_cell_ohm: float = 1.84e-3
    tau1_s: float = 1.5

    # Practical OCV window used for inversion/normalization.
    ocv_cell_min_v: float = 3.0
    ocv_cell_max_v: float = 4.2

    # Polynomial extracted from Samsung datasheet: SOC(OCV) = -209.9*OCV^3 + 2194.8*OCV^2 - 2194.8*OCV + 8133.1
    soc_poly_a3: float = -209.9
    soc_poly_a2: float = 2194.8
    soc_poly_a1: float = -2194.8
    soc_poly_a0: float = 8133.1

    @property
    def q_coulomb(self) -> float:
        return self.capacity_ah * 3600.0

    @property
    def r0_pack_ohm(self) -> float:
        return self.r0_cell_ohm * self.cells_series

    @property
    def r1_pack_ohm(self) -> float:
        return self.r1_cell_ohm * self.cells_series

    @property
    def dcir_pack_ohm(self) -> float:
        return self.dcir_cell_ohm * self.cells_series

    @property
    def c1_pack_f(self) -> float:
        return self.tau1_s / self.r1_pack_ohm


class OCVLookup:

    def __init__(self, params: BatteryParams, grid_points: int = 2001) -> None:
        self.params = params
        self.ocv_grid = [
            params.ocv_cell_min_v
            + i * (params.ocv_cell_max_v - params.ocv_cell_min_v) / (grid_points - 1)
            for i in range(grid_points)
        ]
        self.soc_raw_grid = [self.soc_raw(v) for v in self.ocv_grid]
        self.soc_raw_min = self.soc_raw_grid[0]
        self.soc_raw_max = self.soc_raw_grid[-1]
        self.soc_norm_grid = [self.normalize_raw_soc(x) for x in self.soc_raw_grid]
        self.dv_dsoc_grid = self._build_dv_dsoc_grid()

    def soc_raw(self, ocv_cell_v: float) -> float:
        p = self.params
        return (
            p.soc_poly_a3 * ocv_cell_v ** 3
            + p.soc_poly_a2 * ocv_cell_v ** 2
            + p.soc_poly_a1 * ocv_cell_v
            + p.soc_poly_a0
        )

    def normalize_raw_soc(self, soc_raw: float) -> float:
        span = self.soc_raw_max - self.soc_raw_min
        if span <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (soc_raw - self.soc_raw_min) / span))

    def soc_from_cell_ocv(self, ocv_cell_v: float) -> float:
        return self.normalize_raw_soc(self.soc_raw(ocv_cell_v))

    def raw_soc_from_cell_ocv(self, ocv_cell_v: float) -> float:
        return self.soc_raw(ocv_cell_v)

    def cell_ocv_from_soc(self, soc: float) -> float:
        soc = max(0.0, min(1.0, soc))
        return self._interp(self.soc_norm_grid, self.ocv_grid, soc)

    def pack_ocv_from_soc(self, soc: float) -> float:
        return self.cell_ocv_from_soc(soc) * self.params.cells_series

    def raw_soc_percent_like(self, soc: float) -> float:
        ocv = self.cell_ocv_from_soc(soc)
        return self.raw_soc_from_cell_ocv(ocv)

    def dpack_ocv_dsoc(self, soc: float) -> float:
        soc = max(0.0, min(1.0, soc))
        dv_dsoc_cell = self._interp(self.soc_norm_grid, self.dv_dsoc_grid, soc)
        return dv_dsoc_cell * self.params.cells_series

    def _build_dv_dsoc_grid(self) -> List[float]:
        dv_dsoc: List[float] = []
        x = self.soc_norm_grid
        y = self.ocv_grid
        n = len(x)
        for i in range(n):
            if i == 0:
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
            elif i == n - 1:
                dx = x[i] - x[i - 1]
                dy = y[i] - y[i - 1]
            else:
                dx = x[i + 1] - x[i - 1]
                dy = y[i + 1] - y[i - 1]
            dv_dsoc.append(dy / dx if abs(dx) > 1e-12 else 0.0)
        return dv_dsoc

    @staticmethod
    def _interp(xs: List[float], ys: List[float], xq: float) -> float:
        if xq <= xs[0]:
            return ys[0]
        if xq >= xs[-1]:
            return ys[-1]
        lo = 0
        hi = len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= xq:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if abs(x1 - x0) < 1e-12:
            return y0
        return y0 + (xq - x0) * (y1 - y0) / (x1 - x0)


class BatteryPackModel:
    def __init__(self, params: BatteryParams | None = None) -> None:
        self.params = params or BatteryParams()
        self.ocv = OCVLookup(self.params)
        self.soc = self.ocv.soc_from_cell_ocv(self.params.initial_cell_voltage_v)
        self.v1 = 0.0  # pack-level polarization voltage
        self.time_s = 0.0

    def reset(self) -> None:
        self.soc = self.ocv.soc_from_cell_ocv(self.params.initial_cell_voltage_v)
        self.v1 = 0.0
        self.time_s = 0.0

    def terminal_voltage(self, current_a: float) -> float:
        ocv_pack = self.ocv.pack_ocv_from_soc(self.soc)
        return ocv_pack - current_a * self.params.r0_pack_ohm - self.v1

    def step(self, current_a: float, dt_s: float) -> Dict[str, float]:
        dt_s = max(dt_s, 1e-9)
        alpha = math.exp(-dt_s / self.params.tau1_s)
        self.soc = max(0.0, min(1.0, self.soc - (current_a * dt_s) / self.params.q_coulomb))
        self.v1 = alpha * self.v1 + self.params.r1_pack_ohm * (1.0 - alpha) * current_a
        self.time_s += dt_s

        ocv_pack = self.ocv.pack_ocv_from_soc(self.soc)
        terminal_v = ocv_pack - current_a * self.params.r0_pack_ohm - self.v1
        power_w = terminal_v * current_a
        return {
            "time_s": self.time_s,
            "current_a": current_a,
            "soc": self.soc,
            "soc_percent": self.soc * 100.0,
            "soc_raw_from_poly": self.ocv.raw_soc_percent_like(self.soc),
            "ocv_pack_v": ocv_pack,
            "terminal_v": terminal_v,
            "v1_pack_v": self.v1,
            "power_w": power_w,
        }


def generate_drive_profile(duration_s: int, cycle_name: str = "heavy") -> List[float]:
    """Positive current = discharge."""
    cycle_name = cycle_name.lower()
    profile: List[float] = []
    for t in range(duration_s):
        u = t % 60
        if cycle_name == "transfer":
            if u < 6:
                current = 8.5
            elif u < 45:
                current = 6.2
            elif u < 51:
                current = 3.5
            else:
                current = 1.2
        elif cycle_name == "precision":
            u = t % 8
            if u < 2:
                current = 15.5
            elif u < 3:
                current = 12.5
            elif u < 4:
                current = 8.0
            else:
                current = 9.0
        else:  # Cycle used in paper 
            if u < 4:
                current = 20.5
            elif u < 10:
                current = 18.8
            elif u < 14:
                current = 16.0
            elif u < 18:
                current = 15.0
            elif u < 26:
                current = 18.8
            else:
                current = 12.0
        profile.append(current)
    return profile


def write_mock_measurements_csv(
    output_csv: str | Path,
    duration_s: int = 900,
    cycle_name: str = "heavy",
    current_noise_std_a: float = 0.35,
    voltage_noise_std_v: float = 0.85,
    seed: int = 42,
) -> Path:
    import random

    rng = random.Random(seed)
    model = BatteryPackModel()
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "cycle_name",
                "current_true_a",
                "current_meas_a",
                "terminal_true_v",
                "terminal_meas_v",
                "soc_true_percent",
                "ocv_pack_v",
                "v1_pack_v",
                "power_true_w",
            ],
        )
        writer.writeheader()
        for t, current_true in enumerate(generate_drive_profile(duration_s, cycle_name)):
            state = model.step(current_true, 1.0)
            current_meas = current_true + rng.gauss(0.0, current_noise_std_a)
            terminal_meas = state["terminal_v"] + rng.gauss(0.0, voltage_noise_std_v)
            writer.writerow(
                {
                    "time_s": int(state["time_s"]),
                    "cycle_name": cycle_name,
                    "current_true_a": round(current_true, 6),
                    "current_meas_a": round(current_meas, 6),
                    "terminal_true_v": round(state["terminal_v"], 6),
                    "terminal_meas_v": round(terminal_meas, 6),
                    "soc_true_percent": round(state["soc_percent"], 6),
                    "ocv_pack_v": round(state["ocv_pack_v"], 6),
                    "v1_pack_v": round(state["v1_pack_v"], 6),
                    "power_true_w": round(state["power_w"], 6),
                }
            )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 1 Hz mock battery measurements for the EKF.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--duration", type=int, default=900, help="Simulation duration in seconds.")
    parser.add_argument(
        "--cycle",
        choices=["transfer", "precision", "heavy"],
        default="heavy",
        help="Mock UGV cycle.",
    )
    parser.add_argument("--current-noise", type=float, default=0.15, help="Current noise std-dev in A.")
    parser.add_argument("--voltage-noise", type=float, default=0.05, help="Voltage noise std-dev in V.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    path = write_mock_measurements_csv(
        output_csv=args.output,
        duration_s=args.duration,
        cycle_name=args.cycle,
        current_noise_std_a=args.current_noise,
        voltage_noise_std_v=args.voltage_noise,
        seed=args.seed,
    )
    print(f"Wrote mock measurements to {path}")

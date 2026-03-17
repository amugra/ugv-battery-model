from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from battery_model import BatteryParams, OCVLookup


@dataclass
class EKFConfig:
    process_noise_soc: float = 1e-7
    process_noise_v1: float = 2e-4
    meas_noise_voltage: float = 0.15 ** 2
    init_var_soc: float = 5e-3
    init_var_v1: float = 0.05 ** 2


class BatteryEKF:
    def __init__(self, params: BatteryParams | None = None, config: EKFConfig | None = None) -> None:
        self.params = params or BatteryParams()
        self.config = config or EKFConfig()
        self.ocv = OCVLookup(self.params)

        self.x_soc = self.ocv.soc_from_cell_ocv(self.params.initial_cell_voltage_v)
        self.x_v1 = 0.0
        self.P = [
            [self.config.init_var_soc, 0.0],
            [0.0, self.config.init_var_v1],
        ]
        self.last_terminal_est_v = self.ocv.pack_ocv_from_soc(self.x_soc)
        self.last_residual_v = 0.0
        self.time_s = 0.0

    def predict(self, current_a: float, dt_s: float) -> None:
        dt_s = max(dt_s, 1e-9)
        alpha = math.exp(-dt_s / self.params.tau1_s)

        # State prediction
        self.x_soc = max(0.0, min(1.0, self.x_soc - (current_a * dt_s) / self.params.q_coulomb))
        self.x_v1 = alpha * self.x_v1 + self.params.r1_pack_ohm * (1.0 - alpha) * current_a

        # Jacobian A
        A = [[1.0, 0.0], [0.0, alpha]]
        Q = [[self.config.process_noise_soc, 0.0], [0.0, self.config.process_noise_v1]]
        self.P = self._mat_add(self._mat_mul(self._mat_mul(A, self.P), self._mat_transpose(A)), Q)
        self.time_s += dt_s

    def update(self, terminal_voltage_meas_v: float, current_a: float) -> Dict[str, float]:
        ocv_pack_v = self.ocv.pack_ocv_from_soc(self.x_soc)
        d_ocv_dsoc = self.ocv.dpack_ocv_dsoc(self.x_soc)
        terminal_est_v = ocv_pack_v - current_a * self.params.r0_pack_ohm - self.x_v1

        H = [[d_ocv_dsoc, -1.0]]
        R = [[self.config.meas_noise_voltage]]

        y = terminal_voltage_meas_v - terminal_est_v
        S = self._mat_add(self._mat_mul(self._mat_mul(H, self.P), self._mat_transpose(H)), R)
        K = self._mat_mul(self._mat_mul(self.P, self._mat_transpose(H)), self._mat_inv_1x1(S))

        self.x_soc = max(0.0, min(1.0, self.x_soc + K[0][0] * y))
        self.x_v1 = self.x_v1 + K[1][0] * y

        I = [[1.0, 0.0], [0.0, 1.0]]
        KH = self._mat_mul(K, H)
        self.P = self._mat_mul(self._mat_sub(I, KH), self.P)

        self.last_terminal_est_v = self.ocv.pack_ocv_from_soc(self.x_soc) - current_a * self.params.r0_pack_ohm - self.x_v1
        self.last_residual_v = y

        return self.get_state(current_a)

    def get_state(self, current_a: float) -> Dict[str, float]:
        ocv_pack_v = self.ocv.pack_ocv_from_soc(self.x_soc)
        terminal_est_v = ocv_pack_v - current_a * self.params.r0_pack_ohm - self.x_v1
        power_est_w = terminal_est_v * current_a
        dcir_drop_est_v = current_a * self.params.dcir_pack_ohm
        return {
            "time_s": self.time_s,
            "soc_est": self.x_soc,
            "soc_est_percent": self.x_soc * 100.0,
            "soc_raw_from_poly": self.ocv.raw_soc_percent_like(self.x_soc),
            "ocv_pack_est_v": ocv_pack_v,
            "terminal_est_v": terminal_est_v,
            "v1_est_v": self.x_v1,
            "power_est_w": power_est_w,
            "dcir_drop_est_v": dcir_drop_est_v,
            "innovation_v": self.last_residual_v,
            "var_soc": self.P[0][0],
            "var_v1": self.P[1][1],
        }

    @staticmethod
    def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

    @staticmethod
    def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

    @staticmethod
    def _mat_transpose(A: List[List[float]]) -> List[List[float]]:
        return [list(x) for x in zip(*A)]

    @staticmethod
    def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        rows = len(A)
        cols = len(B[0])
        inner = len(B)
        out = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                s = 0.0
                for k in range(inner):
                    s += A[i][k] * B[k][j]
                out[i][j] = s
        return out

    @staticmethod
    def _mat_inv_1x1(A: List[List[float]]) -> List[List[float]]:
        if abs(A[0][0]) < 1e-12:
            raise ZeroDivisionError("Singular 1x1 matrix in EKF update.")
        return [[1.0 / A[0][0]]]


def run_ekf_from_csv(input_csv: str | Path, output_csv: str | Path, dt_s: float = 1.0) -> Path:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    ekf = BatteryEKF()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="") as fin, output_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "time_s",
                "current_meas_a",
                "terminal_meas_v",
                "soc_est_percent",
                "soc_raw_from_poly",
                "ocv_pack_est_v",
                "terminal_est_v",
                "v1_est_v",
                "power_est_w",
                "dcir_drop_est_v",
                "innovation_v",
                "var_soc",
                "var_v1",
            ],
        )
        writer.writeheader()

        for row in reader:
            current_a = float(row["current_meas_a"])
            terminal_meas_v = float(row["terminal_meas_v"])
            ekf.predict(current_a=current_a, dt_s=dt_s)
            state = ekf.update(terminal_voltage_meas_v=terminal_meas_v, current_a=current_a)
            writer.writerow(
                {
                    "time_s": int(state["time_s"]),
                    "current_meas_a": round(current_a, 6),
                    "terminal_meas_v": round(terminal_meas_v, 6),
                    "soc_est_percent": round(state["soc_est_percent"], 6),
                    "soc_raw_from_poly": round(state["soc_raw_from_poly"], 6),
                    "ocv_pack_est_v": round(state["ocv_pack_est_v"], 6),
                    "terminal_est_v": round(state["terminal_est_v"], 6),
                    "v1_est_v": round(state["v1_est_v"], 6),
                    "power_est_w": round(state["power_est_w"], 6),
                    "dcir_drop_est_v": round(state["dcir_drop_est_v"], 6),
                    "innovation_v": round(state["innovation_v"], 6),
                    "var_soc": round(state["var_soc"], 10),
                    "var_v1": round(state["var_v1"], 10),
                }
            )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the battery EKF from a CSV containing current and terminal voltage.")
    parser.add_argument("--input", required=True, help="Input CSV with current_meas_a and terminal_meas_v columns.")
    parser.add_argument("--output", required=True, help="Output CSV for EKF statistics at 1 Hz.")
    parser.add_argument("--dt", type=float, default=1.0, help="Estimator timestep in seconds.")
    args = parser.parse_args()

    out = run_ekf_from_csv(input_csv=args.input, output_csv=args.output, dt_s=args.dt)
    print(f"Wrote EKF output to {out}")

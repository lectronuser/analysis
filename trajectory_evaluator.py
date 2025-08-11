#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.backends.backend_pdf import PdfPages


class TrajectoryEvaluator:
    def __init__(self, data_dir=Path.home() / "data", default_file_index=1):
        self.data_dir = data_dir
        self.default_file_index = default_file_index
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        # Configuration
        self.time_column = "timestamp"
        self.px4_prefix = "PX4 Pose"
        self.vio_prefix = "VIO Pose"
        self.px4_vel_prefix = "PX4 Vel"
        self.vio_vel_prefix = "VIO Vel"
        self.rpe_delta = 1  # Δt = 1 index (e.g., 100ms if 10Hz)
        self.period = 0.1  # 100ms
        
        # Initialize attributes that will be set later
        self.df = None
        self.px4_xyz = None
        self.vio_xyz = None
        self.vio_aligned = None
        self.errors = None
        self.rpe = None
        self.output_dir = None
        self.result_output_file = None
        self.pdf_filename = None

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Evaluate trajectory accuracy between PX4 and VIO.")
        parser.add_argument("index", nargs="?", default=str(self.default_file_index),
                            help="Index of the pose CSV file (e.g., 1 for pose_1.csv)")
        args = parser.parse_args()
        return args

    def setup_paths(self, file_index):
        csv_file = self.data_dir / f"pose_{file_index}.csv"
        
        self.output_dir = os.path.join(self.data_dir, "output", self.today_str, "Reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.result_output_file = os.path.join(self.output_dir, f"{file_index}.csv")
        self.pdf_filename = str(self.result_output_file).replace(".csv", "_report.pdf")
        return csv_file

    def load_data(self, csv_file):
        with open(csv_file, "r") as f:
            lines = f.readlines()[:-1]

        self.df = pd.read_csv(StringIO("".join(lines)))

    def validate_columns(self):
        required_columns = [
            f"{self.px4_prefix} X", f"{self.px4_prefix} Y", f"{self.px4_prefix} Z",
            f"{self.vio_prefix} X", f"{self.vio_prefix} Y", f"{self.vio_prefix} Z"
        ]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            print(f"Missing columns in CSV: {missing}")
            sys.exit(1)

    def extract_trajectories(self):
        self.px4_xyz = self.df[[f"{self.px4_prefix} X", f"{self.px4_prefix} Y", f"{self.px4_prefix} Z"]].values
        self.vio_xyz = self.df[[f"{self.vio_prefix} X", f"{self.vio_prefix} Y", f"{self.vio_prefix} Z"]].values

    def align_trajectories(self):
        vio_mean = np.mean(self.vio_xyz, axis=0)
        gt_mean = np.mean(self.px4_xyz, axis=0)
        vio_centered = self.vio_xyz - vio_mean
        gt_centered = self.px4_xyz - gt_mean

        H = vio_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T

        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T

        t_align = gt_mean - R_align @ vio_mean
        self.vio_aligned = (R_align @ self.vio_xyz.T).T + t_align

    def compute_ate(self):
        self.errors = np.linalg.norm(self.vio_aligned - self.px4_xyz, axis=1)
        self.ate_rmse = np.sqrt(np.mean(self.errors**2))
        self.ate_mean = np.mean(self.errors)
        self.ate_median = np.median(self.errors)
        self.ate_std = np.std(self.errors)

    def compute_rpe(self):
        rpe = []
        for i in range(len(self.px4_xyz) - self.rpe_delta):
            delta_px4 = self.px4_xyz[i + self.rpe_delta] - self.px4_xyz[i]
            delta_vio = self.vio_aligned[i + self.rpe_delta] - self.vio_aligned[i]
            rpe.append(np.linalg.norm(delta_px4 - delta_vio))
        self.rpe = np.array(rpe)

        self.rpe_rmse = np.sqrt(np.mean(self.rpe**2))
        self.rpe_mean = np.mean(self.rpe)
        self.rpe_median = np.median(self.rpe)
        self.rpe_std = np.std(self.rpe)

    def compute_distance_stats(self):
        # Compute distance traveled
        self.px4_distances = np.linalg.norm(np.diff(self.px4_xyz[:, :2], axis=0), axis=1)
        self.px4_total_distance = np.sum(self.px4_distances)

        self.vio_distances = np.linalg.norm(np.diff(self.vio_xyz[:, :2], axis=0), axis=1)
        self.vio_total_distance = np.sum(self.vio_distances)

        self.total_distance_difference = np.abs(self.px4_total_distance - self.vio_total_distance)
        self.total_distance_difference_range = self.total_distance_difference / self.px4_total_distance * 100

        # Compute height statistics
        self.max_px4_height = self.px4_xyz[:, 2].max()
        self.min_px4_height = self.px4_xyz[:, 2].min()
        self.mean_px4_height = self.px4_xyz[:, 2].mean()

        self.max_vio_height = self.vio_xyz[:, 2].max()
        self.min_vio_height = self.vio_xyz[:, 2].min()
        self.mean_vio_height = self.vio_xyz[:, 2].mean()

        # Time consumption
        self.rows_size = self.df.shape[0]
        self.time_consumption = self.rows_size * self.period

    def compute_velocity_stats(self):
        required_vel_columns = [
            f"{self.px4_vel_prefix} X", f"{self.px4_vel_prefix} Y", f"{self.px4_vel_prefix} Z",
            f"{self.vio_vel_prefix} X", f"{self.vio_vel_prefix} Y", f"{self.vio_vel_prefix} Z"
        ]
        self.missing_vel = [col for col in required_vel_columns if col not in self.df.columns]
        
        if self.missing_vel:
            print(f"Missing velocity columns in CSV: {self.missing_vel}")
            return False
        
        px4_vel = self.df[[f"{self.px4_vel_prefix} X", f"{self.px4_vel_prefix} Y", f"{self.px4_vel_prefix} Z"]].values
        vio_vel = self.df[[f"{self.vio_vel_prefix} X", f"{self.vio_vel_prefix} Y", f"{self.vio_vel_prefix} Z"]].values

        px4_speeds = np.linalg.norm(px4_vel, axis=1)
        vio_speeds = np.linalg.norm(vio_vel, axis=1)

        self.mean_px4_speed = np.mean(px4_speeds)
        self.max_px4_speed = np.max(px4_speeds)
        self.min_px4_speed = np.min(px4_speeds)
        self.std_px4_speed = np.std(px4_speeds)

        self.mean_vio_speed = np.mean(vio_speeds)
        self.max_vio_speed = np.max(vio_speeds)
        self.min_vio_speed = np.min(vio_speeds)
        self.std_vio_speed = np.std(vio_speeds)
        
        return True

    def get_info(self):
        dev_model = input("Device Model: ")
        mission_input = input("Mission: ")
        mission_output = input("Output: ")
        mission_explanation = input("Explanation: ").replace("\\n", "\n")

        text = f"""
        Device Model: {dev_model},
        Mission:  {mission_input},
        Output:  {mission_output},
        Explanation: {mission_explanation}
        """

        return text

    def generate_report_text(self):
        text= self.get_info()

        text += f"""
        Time consumption: {self.time_consumption:.2f} seconds

        == ATE (Absolute Trajectory Error) ==
        RMSE   : {self.ate_rmse:.4f} m
        MEAN   : {self.ate_mean:.4f} m
        MEDIAN : {self.ate_median:.4f} m
        STD    : {self.ate_std:.4f} m

        == RPE (Relative Pose Error, Δt = 100ms) ==
        RMSE   : {self.rpe_rmse:.4f} m
        MEAN   : {self.rpe_mean:.4f} m
        MEDIAN : {self.rpe_median:.4f} m
        STD    : {self.rpe_std:.4f} m

        == Total Distance Traveled on X-Y Plane ==
        PX4 Total Distance: {self.px4_total_distance:.2f} meters
        VIO Total Distance: {self.vio_total_distance:.2f} meters

        Total Distance Difference: {self.total_distance_difference:.2f} meters
        Total Distance Difference Range: {self.total_distance_difference_range:.2f} %

        == PX4 Height Statistics ==
        Max height of PX4 estimate is {self.max_px4_height:.2f}
        Min height of PX4 estimate is {self.min_px4_height:.2f}
        Mean height of PX4 estimate is {self.mean_px4_height:.2f}

        == VIO Height Statistics ==
        Max height of  VIO estimate is {self.max_vio_height:.2f}
        Min height of  VIO estimate is {self.min_vio_height:.2f}
        Mean height of VIO estimate is {self.mean_vio_height:.2f}
        """

        if not self.missing_vel:
            text += f"""\n
        == PX4 Velocity Statistics ==
        Mean speed: {self.mean_px4_speed:.2f} m/s
        Max speed : {self.max_px4_speed:.2f} m/s
        Min speed : {self.min_px4_speed:.2f} m/s
        STD speed : {self.std_px4_speed:.2f} m/s

        == VIO Velocity Statistics ==
        Mean speed: {self.mean_vio_speed:.2f} m/s
        Max speed : {self.max_vio_speed:.2f} m/s
        Min speed : {self.min_vio_speed:.2f} m/s
        STD speed : {self.std_vio_speed:.2f} m/s
        """

        return text

    def print_summary(self):
        print("\n========= SUMMARY =========\n")
        print(f"Time consumption: {self.time_consumption:.2f} seconds\n")

        print("\n== ATE (Absolute Trajectory Error) ==")
        print(f"RMSE   : {self.ate_rmse:.4f} m")
        print(f"MEAN   : {self.ate_mean:.4f} m")
        print(f"MEDIAN : {self.ate_median:.4f} m")
        print(f"STD    : {self.ate_std:.4f} m")

        print("\n== RPE (Relative Pose Error, Δt = 100ms) ==")
        print(f"RMSE   : {self.rpe_rmse:.4f} m")
        print(f"MEAN   : {self.rpe_mean:.4f} m")
        print(f"MEDIAN : {self.rpe_median:.4f} m")
        print(f"STD    : {self.rpe_std:.4f} m")

        print("\n== Total Distance Traveled on X-Y Plane ==")
        print(f"PX4 Total Distance: {self.px4_total_distance:.2f} meters")
        print(f"VIO Total Distance: {self.vio_total_distance:.2f} meters")
        print(f"Total Distance Difference: {self.total_distance_difference:.2f} meters")
        print(f"Total Distance Difference Range: {self.total_distance_difference_range:.2f} %\n")

        print(f"Max height of PX4 estimate is {self.max_px4_height:.2f}")
        print(f"Min height of PX4 estimate is {self.min_px4_height:.2f}")
        print(f"Mean height of PX4 estimate is {self.mean_px4_height:.2f}\n")

        print(f"Max height of VIO estimate is {self.max_vio_height:.2f}")
        print(f"Min height of VIO estimate is {self.min_vio_height:.2f}")
        print(f"Mean height of VIO estimate is {self.mean_vio_height:.2f}\n")

        if not self.missing_vel:
            print("== PX4 Velocity Statistics ==")
            print(f"Mean speed: {self.mean_px4_speed:.2f} m/s")
            print(f"Max speed : {self.max_px4_speed:.2f} m/s")
            print(f"Min speed : {self.min_px4_speed:.2f} m/s")
            print(f"STD speed : {self.std_px4_speed:.2f} m/s\n")

            print("== VIO Velocity Statistics ==")
            print(f"Mean speed: {self.mean_vio_speed:.2f} m/s")
            print(f"Max speed : {self.max_vio_speed:.2f} m/s")
            print(f"Min speed : {self.min_vio_speed:.2f} m/s")
            print(f"STD speed : {self.std_vio_speed:.2f} m/s\n")

    def generate_pdf_report(self):
        with PdfPages(self.pdf_filename) as pdf:
            # Trajectory Plot
            plt.figure(figsize=(10, 8))
            plt.plot(self.px4_xyz[:, 0], self.px4_xyz[:, 1], label="PX4 (Ground Truth)", linewidth=2)
            plt.plot(self.vio_aligned[:, 0], self.vio_aligned[:, 1], label="VIO (Estimated)", linewidth=2)
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title("2D Trajectory Comparison (X-Y Plane)")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Summary Text
            fig_text = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.01, 0.99, self.generate_report_text(), va='top', ha='left', fontsize=12, family='monospace')
            pdf.savefig(fig_text)
            plt.close()

    def run(self):
        args = self.parse_arguments()
        csv_file = self.setup_paths(args.index)
        
        self.load_data(csv_file)
        self.validate_columns()
        self.extract_trajectories()
        self.align_trajectories()
        
        self.compute_ate()
        self.compute_rpe()
        self.compute_distance_stats()
        has_velocity = self.compute_velocity_stats()
        
        self.print_summary()
        self.generate_pdf_report()
        
        print(f"PDF report saved as: {self.pdf_filename}")


if __name__ == "__main__":
    evaluator = TrajectoryEvaluator()
    evaluator.run()
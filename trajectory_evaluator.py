#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages

# ==== Configuration ====
data_dir = Path.home() / "data"
default_file_index = 1

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate trajectory accuracy between PX4 and VIO.")
parser.add_argument("index", nargs="?", default=str(default_file_index),
                    help="Index of the pose CSV file (e.g., 1 for pose_1.csv)")
args = parser.parse_args()
csv_file = data_dir / f"pose_{args.index}.csv"

time_column = "timestamp"
px4_prefix = "PX4 Pose"
vio_prefix = "VIO Pose"
rpe_delta = 1  # Δt = 1 index (e.g., 100ms if 10Hz)

# ==== Load CSV ====
with open(csv_file, "r") as f:
    lines = f.readlines()[:-1]

df = pd.read_csv(StringIO("".join(lines)))

# ==== Validate Columns ====
required_columns = [f"{px4_prefix} X", f"{px4_prefix} Y", f"{px4_prefix} Z",
                    f"{vio_prefix} X", f"{vio_prefix} Y", f"{vio_prefix} Z"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns in CSV: {missing}")
    sys.exit(1)

# ==== Extract and Align ====
px4_xyz = df[[f"{px4_prefix} X", f"{px4_prefix} Y", f"{px4_prefix} Z"]].values
vio_xyz = df[[f"{vio_prefix} X", f"{vio_prefix} Y", f"{vio_prefix} Z"]].values

def align_trajectories(vio, gt):
    vio_mean = np.mean(vio, axis=0)
    gt_mean = np.mean(gt, axis=0)
    vio_centered = vio - vio_mean
    gt_centered = gt - gt_mean

    H = vio_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    t_align = gt_mean - R_align @ vio_mean
    vio_aligned = (R_align @ vio.T).T + t_align
    return vio_aligned

vio_aligned = align_trajectories(vio_xyz, px4_xyz)

# ==== Compute ATE ====
errors = np.linalg.norm(vio_aligned - px4_xyz, axis=1)
ate_rmse = np.sqrt(np.mean(errors**2))
ate_mean = np.mean(errors)
ate_median = np.median(errors)
ate_std = np.std(errors)

# ==== Compute RPE ====
rpe = []
for i in range(len(px4_xyz) - rpe_delta):
    delta_px4 = px4_xyz[i + rpe_delta] - px4_xyz[i]
    delta_vio = vio_aligned[i + rpe_delta] - vio_aligned[i]
    rpe.append(np.linalg.norm(delta_px4 - delta_vio))
rpe = np.array(rpe)

rpe_rmse = np.sqrt(np.mean(rpe**2))
rpe_mean = np.mean(rpe)
rpe_median = np.median(rpe)
rpe_std = np.std(rpe)

# ==== Compute Distance Traveled ====
px4_distances = np.linalg.norm(np.diff(px4_xyz[:, :2], axis=0), axis=1)
px4_total_distance = np.sum(px4_distances)

vio_distances = np.linalg.norm(np.diff(vio_xyz[:, :2], axis=0), axis=1)
vio_total_distance = np.sum(vio_distances)

total_distance_difference = np.abs(px4_total_distance - vio_total_distance)
total_distance_difference_range = total_distance_difference / px4_total_distance * 100

# ==== Display Console Summary ====
print("\n== ATE (Absolute Trajectory Error) ==")
print(f"RMSE   : {ate_rmse:.4f} m")
print(f"MEAN   : {ate_mean:.4f} m")
print(f"MEDIAN : {ate_median:.4f} m")
print(f"STD    : {ate_std:.4f} m")

print("\n== RPE (Relative Pose Error, Δt = 100ms) ==")
print(f"RMSE   : {rpe_rmse:.4f} m")
print(f"MEAN   : {rpe_mean:.4f} m")
print(f"MEDIAN : {rpe_median:.4f} m")
print(f"STD    : {rpe_std:.4f} m")

print("\n== Total Distance Traveled on X-Y Plane ==")
print(f"PX4 Total Distance: {px4_total_distance:.2f} meters")
print(f"VIO Total Distance: {vio_total_distance:.2f} meters")
print(f"Total Distance Difference: {total_distance_difference:.2f} meters")
print(f"Total Distance Difference Range: {total_distance_difference_range:.2f} %\n")

# ==== Save Report as PDF ====
pdf_filename = str(csv_file).replace(".csv", "_report.pdf")
with PdfPages(pdf_filename) as pdf:
    # Trajectory Plot
    plt.figure(figsize=(10, 8))
    plt.plot(px4_xyz[:, 0], px4_xyz[:, 1], label="PX4 (Ground Truth)", linewidth=2)
    plt.plot(vio_aligned[:, 0], vio_aligned[:, 1], label="VIO (Estimated)", linewidth=2)
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
    text = f"""
    == ATE (Absolute Trajectory Error) ==
    RMSE   : {ate_rmse:.4f} m
    MEAN   : {ate_mean:.4f} m
    MEDIAN : {ate_median:.4f} m
    STD    : {ate_std:.4f} m

    == RPE (Relative Pose Error, Δt = 100ms) ==
    RMSE   : {rpe_rmse:.4f} m
    MEAN   : {rpe_mean:.4f} m
    MEDIAN : {rpe_median:.4f} m
    STD    : {rpe_std:.4f} m

    == Total Distance Traveled on X-Y Plane ==
    PX4 Total Distance: {px4_total_distance:.2f} meters
    VIO Total Distance: {vio_total_distance:.2f} meters

    Total Distance Difference: {total_distance_difference:.2f} meters
    Total Distance Difference Range: {total_distance_difference_range:.2f} %
    """
    plt.axis('off')
    plt.text(0.01, 0.99, text, va='top', ha='left', fontsize=12, family='monospace')
    pdf.savefig(fig_text)
    plt.close()

print(f"PDF report saved as: {pdf_filename}")

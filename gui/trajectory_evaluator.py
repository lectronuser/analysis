#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

class TrajectoryEvaluator:
    def __init__(self):
        """Initialize with default parameters"""
        self._input_path = os.path.expanduser("~/data")
        self._file_index = 1
        self._period = 0.1  # 100ms
        self._rpe_delta = 1  # Δt = 1 index
        self._output_folder = None
        self._output_file = None
        
        self._time_column = "timestamp"
        self._px4_prefix = "PX4 Pose"
        self._vio_prefix = "VIO Pose"
        self._px4_vel_prefix = "PX4 Vel"
        self._vio_vel_prefix = "VIO Vel"

        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        self.setup_output()

    @property
    def input_path(self):
        return self._input_path
    
    @input_path.setter
    def input_path(self, value):
        self._input_path = value
        self.setup_output()
    
    @property
    def file_index(self):
        return self._file_index
    
    @file_index.setter
    def file_index(self, value):
        self._file_index = int(value)
    
    @property
    def period(self):
        return self._period
    
    @period.setter
    def period(self, value):
        self._period = float(value)
    
    @property
    def rpe_delta(self):
        return self._rpe_delta
    
    @rpe_delta.setter
    def rpe_delta(self, value):
        self._rpe_delta = int(value)

    def setup_output(self):
        if self._input_path:
            self._output_folder = os.path.join(self._input_path, "output", "report")
            os.makedirs(self._output_folder, exist_ok=True)
            today_str = datetime.today().strftime("%Y_%m_%d")
            self._output_file = os.path.join(
                self._output_folder, 
                f"pose_{self._file_index}_report_{today_str}.pdf"
            )

    def load_data(self):
        csv_file = os.path.join(self._input_path, f"pose_{self._file_index}.csv")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        required_columns = [
            f"{self._px4_prefix} X", f"{self._px4_prefix} Y", f"{self._px4_prefix} Z",
            f"{self._vio_prefix} X", f"{self._vio_prefix} Y", f"{self._vio_prefix} Z"
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df

    def align_trajectories(self, source, target):
        source_mean = np.mean(source, axis=0)
        target_mean = np.mean(target, axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean

        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T

        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T

        t_align = target_mean - R_align @ source_mean
        aligned = (R_align @ source.T).T + t_align
        
        return aligned, R_align, t_align

    def calculate_velocity_stats(self, df):
        vel_stats = {}
        required_vel_columns = [
            f"{self._px4_vel_prefix} X", f"{self._px4_vel_prefix} Y", f"{self._px4_vel_prefix} Z",
            f"{self._vio_vel_prefix} X", f"{self._vio_vel_prefix} Y", f"{self._vio_vel_prefix} Z"
        ]
        
        if all(col in df.columns for col in required_vel_columns):
            px4_vel = df[[f"{self._px4_vel_prefix} X", f"{self._px4_vel_prefix} Y", 
                         f"{self._px4_vel_prefix} Z"]].values
            vio_vel = df[[f"{self._vio_vel_prefix} X", f"{self._vio_vel_prefix} Y", 
                         f"{self._vio_vel_prefix} Z"]].values

            for prefix, vel in [('px4', px4_vel), ('vio', vio_vel)]:
                speeds = np.linalg.norm(vel, axis=1)
                vel_stats[f'{prefix}_speed'] = {
                    'mean': np.mean(speeds),
                    'max': np.max(speeds),
                    'min': np.min(speeds),
                    'std': np.std(speeds)
                }
        
        return vel_stats

    def evaluate_trajectory(self, df):
        px4_xyz = df[[f"{self._px4_prefix} X", f"{self._px4_prefix} Y", 
                     f"{self._px4_prefix} Z"]].values
        vio_xyz = df[[f"{self._vio_prefix} X", f"{self._vio_prefix} Y", 
                     f"{self._vio_prefix} Z"]].values

        vio_aligned, R, t = self.align_trajectories(vio_xyz, px4_xyz)

        errors = np.linalg.norm(vio_aligned - px4_xyz, axis=1)
        ate_stats = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'max': np.max(errors),
            'min': np.min(errors)
        }

        rpe = []
        for i in range(len(px4_xyz) - self._rpe_delta):
            delta_px4 = px4_xyz[i + self._rpe_delta] - px4_xyz[i]
            delta_vio = vio_aligned[i + self._rpe_delta] - vio_aligned[i]
            rpe.append(np.linalg.norm(delta_px4 - delta_vio))
        rpe = np.array(rpe)
        
        rpe_stats = {
            'rmse': np.sqrt(np.mean(rpe**2)),
            'mean': np.mean(rpe),
            'median': np.median(rpe),
            'std': np.std(rpe),
            'max': np.max(rpe),
            'min': np.min(rpe)
        }

        px4_distances = np.linalg.norm(np.diff(px4_xyz[:, :2], axis=0), axis=1)
        px4_total_distance = np.sum(px4_distances)

        vio_distances = np.linalg.norm(np.diff(vio_xyz[:, :2], axis=0), axis=1)
        vio_total_distance = np.sum(vio_distances)

        distance_stats = {
            'px4_total': px4_total_distance,
            'vio_total': vio_total_distance,
            'absolute_diff': np.abs(px4_total_distance - vio_total_distance),
            'relative_diff': (np.abs(px4_total_distance - vio_total_distance) / 
                            px4_total_distance * 100)
        }

        height_stats = {
            'px4': {
                'max': px4_xyz[:, 2].max(),
                'min': px4_xyz[:, 2].min(),
                'mean': px4_xyz[:, 2].mean()
            },
            'vio': {
                'max': vio_xyz[:, 2].max(),
                'min': vio_xyz[:, 2].min(),
                'mean': vio_xyz[:, 2].mean()
            }
        }

        time_stats = {
            'duration': df.shape[0] * self._period,
            'data_points': df.shape[0],
            'frequency': 1 / self._period
        }

        vel_stats = self.calculate_velocity_stats(df)

        return {
            'ate': ate_stats,
            'rpe': rpe_stats,
            'distance': distance_stats,
            'height': height_stats,
            'time': time_stats,
            'velocity': vel_stats,
            'transformation': {
                'rotation': R.tolist(),
                'translation': t.tolist()
            },
            'data': {
                'px4': px4_xyz,
                'vio': vio_xyz,
                'vio_aligned': vio_aligned
            }
        }

    def generate_report(self, results, output_file=None):
        if not output_file:
            output_file = self._output_file
        
        with PdfPages(output_file) as pdf:
            fig1 = plt.figure(figsize=(10, 8))
            plt.plot(results['data']['px4'][:, 0], results['data']['px4'][:, 1], 
                    label="PX4 (Ground Truth)", linewidth=2)
            plt.plot(results['data']['vio_aligned'][:, 0], results['data']['vio_aligned'][:, 1], 
                    label="VIO (Estimated)", linewidth=2)
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title("2D Trajectory Comparison (X-Y Plane)")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close()

            fig_text = plt.figure(figsize=(8.5, 11))
            text = self._generate_report_text(results)
            plt.axis('off')
            plt.text(0.01, 0.99, text, va='top', ha='left', 
                    fontsize=12, family='monospace')
            pdf.savefig(fig_text)
            plt.close()

            if sys.platform == "win32":
                os.startfile(output_file)
            elif sys.platform == "darwin":
                subprocess.run(["open", output_file])
            else:
                subprocess.run(["xdg-open", output_file])
                print("bu kısım çalıştır")

        return output_file

    def _generate_report_text(self, results):
        """Generate formatted text for report"""
        text = f"""
        == Time Statistics ==
        Duration      : {results['time']['duration']:.2f} seconds
        Data Points   : {results['time']['data_points']}
        Frequency     : {results['time']['frequency']:.1f} Hz

        == ATE (Absolute Trajectory Error) ==
        RMSE   : {results['ate']['rmse']:.4f} m
        MEAN   : {results['ate']['mean']:.4f} m
        MEDIAN : {results['ate']['median']:.4f} m
        STD    : {results['ate']['std']:.4f} m
        MAX    : {results['ate']['max']:.4f} m
        MIN    : {results['ate']['min']:.4f} m

        == RPE (Relative Pose Error, Δt = {self._rpe_delta*self._period*1000:.0f}ms) ==
        RMSE   : {results['rpe']['rmse']:.4f} m
        MEAN   : {results['rpe']['mean']:.4f} m
        MEDIAN : {results['rpe']['median']:.4f} m
        STD    : {results['rpe']['std']:.4f} m
        MAX    : {results['rpe']['max']:.4f} m
        MIN    : {results['rpe']['min']:.4f} m

        == Distance Traveled (X-Y Plane) ==
        PX4 Total Distance: {results['distance']['px4_total']:.2f} m
        VIO Total Distance: {results['distance']['vio_total']:.2f} m
        Absolute Difference: {results['distance']['absolute_diff']:.2f} m
        Relative Difference: {results['distance']['relative_diff']:.2f} %

        == Height Statistics ==
        PX4 Height:
          MAX : {results['height']['px4']['max']:.2f} m
          MIN : {results['height']['px4']['min']:.2f} m
          MEAN: {results['height']['px4']['mean']:.2f} m

        VIO Height:
          MAX : {results['height']['vio']['max']:.2f} m
          MIN : {results['height']['vio']['min']:.2f} m
          MEAN: {results['height']['vio']['mean']:.2f} m
        """

        if results.get('velocity'):
            text += f"""
        == Velocity Statistics ==
        PX4 Velocity:
          MEAN: {results['velocity']['px4_speed']['mean']:.2f} m/s
          MAX : {results['velocity']['px4_speed']['max']:.2f} m/s
          MIN : {results['velocity']['px4_speed']['min']:.2f} m/s
          STD : {results['velocity']['px4_speed']['std']:.2f} m/s

        VIO Velocity:
          MEAN: {results['velocity']['vio_speed']['mean']:.2f} m/s
          MAX : {results['velocity']['vio_speed']['max']:.2f} m/s
          MIN : {results['velocity']['vio_speed']['min']:.2f} m/s
          STD : {results['velocity']['vio_speed']['std']:.2f} m/s
        """

        return text

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        try:
            df = self.load_data()
            results = self.evaluate_trajectory(df)
            report_file = self.generate_report(results)
            
            return {
                'success': True,
                'results': results,
                'report_file': report_file,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'results': None,
                'report_file': None,
                'error': str(e)
            }
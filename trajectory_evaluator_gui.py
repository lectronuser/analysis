import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, 
                             QGroupBox, QFormLayout, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QSettings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import datetime
import numpy as np
import pandas as pd
from io import StringIO
from scipy.spatial.transform import Rotation as R


class TrajectoryEvaluatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trajectory Evaluator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize settings
        self.settings = QSettings("TrajectoryEvaluator", "AppSettings")
        
        # Initialize attributes
        self.df = None
        self.px4_xyz = None
        self.vio_xyz = None
        self.fus_xyz = None
        self.vio_aligned = None
        self.fus_aligned = None
        self.errors = None
        self.rpe = None
        self.output_dir = None
        self.result_output_file = None
        self.pdf_filename = None
        self.current_csv_path = None
        self.data_dir = Path.home() / "data"  # Base data directory
        
        # Configuration
        self.time_column = "timestamp"
        self.px4_prefix = "PX4 Pose"
        self.vio_prefix = "VIO Pose"
        self.fus_prefix = "FUS Pose"
        self.px4_vel_prefix = "PX4 Vel"
        self.vio_vel_prefix = "VIO Vel"
        self.fus_vel_prefix = "FUS Vel"
        self.rpe_delta = 1
        self.period = 0.1
        
        self.initUI()
        
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Right panel for plots
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select pose CSV file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        
        file_layout.addRow(QLabel("CSV File:"), self.file_path_edit)
        file_layout.addRow(browse_btn)
        file_group.setLayout(file_layout)
        
        # Test information
        info_group = QGroupBox("Test Information")
        info_layout = QFormLayout()
        
        self.test_num_edit = QLineEdit()
        self.device_model_edit = QLineEdit()
        
        info_layout.addRow(QLabel("Test Number:"), self.test_num_edit)
        info_layout.addRow(QLabel("Device Model:"), self.device_model_edit)
        info_group.setLayout(info_layout)
        
        # Mission parameters
        mission_group = QGroupBox("Mission Parameters")
        mission_layout = QFormLayout()
        
        self.distance_edit = QLineEdit()
        self.yaw_edit = QLineEdit()
        self.height_edit = QLineEdit()
        self.max_speed_edit = QLineEdit()
        
        mission_layout.addRow(QLabel("Distance (m):"), self.distance_edit)
        mission_layout.addRow(QLabel("Yaw (deg):"), self.yaw_edit)
        mission_layout.addRow(QLabel("Height (m):"), self.height_edit)
        mission_layout.addRow(QLabel("Max Speed (m/s):"), self.max_speed_edit)
        mission_group.setLayout(mission_layout)
        
        # Mission results
        result_group = QGroupBox("Mission Results")
        result_layout = QFormLayout()
        
        self.actual_distance_edit = QLineEdit()
        self.result_explanation_edit = QTextEdit()
        self.result_explanation_edit.setMaximumHeight(100)
        
        result_layout.addRow(QLabel("Actual Distance (m):"), self.actual_distance_edit)
        result_layout.addRow(QLabel("Explanation:"), self.result_explanation_edit)
        result_group.setLayout(result_layout)
        
        # Report settings
        report_group = QGroupBox("Report Settings")
        report_layout = QFormLayout()
        
        self.report_name_edit = QLineEdit()
        self.report_name_edit.setPlaceholderText("Enter report name (without extension)")
        
        report_layout.addRow(QLabel("Report Name:"), self.report_name_edit)
        report_group.setLayout(report_layout)
        
        # Buttons
        self.evaluate_btn = QPushButton("Evaluate Trajectory")
        self.evaluate_btn.clicked.connect(self.evaluate_trajectory)
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setEnabled(False)
        
        # Add widgets to left layout
        left_layout.addWidget(file_group)
        left_layout.addWidget(info_group)
        left_layout.addWidget(mission_group)
        left_layout.addWidget(result_group)
        left_layout.addWidget(report_group)
        left_layout.addWidget(self.evaluate_btn)
        left_layout.addWidget(self.export_btn)
        left_layout.addStretch()
        
        # Plot area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def browse_file(self):
        # Get last directory from settings or use data directory
        last_dir = self.settings.value("last_dir", str(self.data_dir))
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pose CSV File", last_dir, 
            "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            self.file_path_edit.setText(file_path)
            self.current_csv_path = Path(file_path)
            # Save directory to settings
            self.settings.setValue("last_dir", str(self.current_csv_path.parent))
            
            # Set default report name based on CSV file name
            # report_name = self.current_csv_path.stem.replace("pose_", "report_")
            report_name = self.test_num_edit.text() + "_report"
            self.report_name_edit.setText(report_name)
            
    def load_data(self, csv_file):
        with open(csv_file, "r") as f:
            lines = f.readlines()[:-1]

        self.df = pd.read_csv(StringIO("".join(lines)))
        
    def validate_columns(self):
        required_columns = [
            f"{self.px4_prefix} X", f"{self.px4_prefix} Y", f"{self.px4_prefix} Z",
            f"{self.vio_prefix} X", f"{self.vio_prefix} Y", f"{self.vio_prefix} Z",
            f"{self.fus_prefix} X", f"{self.fus_prefix} Y", f"{self.fus_prefix} Z"
        ]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            QMessageBox.critical(self, "Error", f"Missing columns in CSV: {missing}")
            return False
        return True
    
    def extract_trajectories(self):
        self.px4_xyz = self.df[[f"{self.px4_prefix} X", f"{self.px4_prefix} Y", f"{self.px4_prefix} Z"]].values
        self.vio_xyz = self.df[[f"{self.vio_prefix} X", f"{self.vio_prefix} Y", f"{self.vio_prefix} Z"]].values
        self.fus_xyz = self.df[[f"{self.fus_prefix} X", f"{self.fus_prefix} Y", f"{self.fus_prefix} Z"]].values
        
    def align_trajectories(self):
        # Align VIO to PX4
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

        # Align FUS to PX4
        fus_mean = np.mean(self.fus_xyz, axis=0)
        fus_centered = self.fus_xyz - fus_mean
        
        H = fus_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T

        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T

        t_align = gt_mean - R_align @ fus_mean
        self.fus_aligned = (R_align @ self.fus_xyz.T).T + t_align
        
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
        # Track which velocity data is available
        self.has_px4_vel = False
        self.has_vio_vel = False
        self.has_fus_vel = False
        
        # Try to compute PX4 velocity stats
        px4_cols = [f"{self.px4_vel_prefix} X", f"{self.px4_vel_prefix} Y", f"{self.px4_vel_prefix} Z"]
        if all(col in self.df.columns for col in px4_cols):
            px4_vel = self.df[px4_cols].values
            px4_speeds = np.linalg.norm(px4_vel, axis=1)
            self.mean_px4_speed = np.mean(px4_speeds)
            self.max_px4_speed = np.max(px4_speeds)
            self.min_px4_speed = np.min(px4_speeds)
            self.std_px4_speed = np.std(px4_speeds)
            self.has_px4_vel = True
            
        # Try to compute VIO velocity stats
        vio_cols = [f"{self.vio_vel_prefix} X", f"{self.vio_vel_prefix} Y", f"{self.vio_vel_prefix} Z"]
        if all(col in self.df.columns for col in vio_cols):
            vio_vel = self.df[vio_cols].values
            vio_speeds = np.linalg.norm(vio_vel, axis=1)
            self.mean_vio_speed = np.mean(vio_speeds)
            self.max_vio_speed = np.max(vio_speeds)
            self.min_vio_speed = np.min(vio_speeds)
            self.std_vio_speed = np.std(vio_speeds)
            self.has_vio_vel = True
            
        # Try to compute FUS velocity stats
        fus_cols = [f"{self.fus_vel_prefix} X", f"{self.fus_vel_prefix} Y", f"{self.fus_vel_prefix} Z"]
        if all(col in self.df.columns for col in fus_cols):
            fus_vel = self.df[fus_cols].values
            fus_speeds = np.linalg.norm(fus_vel, axis=1)
            self.mean_fus_speed = np.mean(fus_speeds)
            self.max_fus_speed = np.max(fus_speeds)
            self.min_fus_speed = np.min(fus_speeds)
            self.std_fus_speed = np.std(fus_speeds)
            self.has_fus_vel = True
            
        # Report missing velocity data but don't prevent plotting what we have
        missing_systems = []
        if not self.has_px4_vel:
            missing_systems.append("PX4")
        if not self.has_vio_vel:
            missing_systems.append("VIO")
        if not self.has_fus_vel:
            missing_systems.append("FUS")
            
        if missing_systems:
            QMessageBox.warning(self, "Warning", 
                              f"Missing velocity data for: {', '.join(missing_systems)}\n\n"
                              "Only available velocity data will be shown.")
        
        return True
        
    def plot_trajectory(self):
        self.figure.clear()
        
        # Create subplot for trajectories
        ax1 = self.figure.add_subplot(211)
        
        # Plot raw trajectories (no alignment)
        ax1.plot(self.px4_xyz[:, 0], self.px4_xyz[:, 1], label="PX4 (Ground Truth)", linewidth=2)
        ax1.plot(self.vio_xyz[:, 0], self.vio_xyz[:, 1], label="VIO (Raw)", linewidth=2)
        ax1.plot(self.fus_xyz[:, 0], self.fus_xyz[:, 1], label="FUS (Raw)", linewidth=2, color='purple')
        
        # mark start / end points for PX4
        if self.px4_xyz is not None and len(self.px4_xyz) > 0:
            px4_start = self.px4_xyz[0, :2]
            px4_end = self.px4_xyz[-1, :2]
            ax1.scatter(px4_start[0], px4_start[1], c='green', marker='o', s=80, zorder=5, label='PX4 Start')
            ax1.scatter(px4_end[0], px4_end[1], c='darkgreen', marker='X', s=80, zorder=5, label='PX4 End')
            ax1.annotate('PX4 Start', xy=(px4_start[0], px4_start[1]), xytext=(5, 5),
                        textcoords='offset points', color='green', fontsize=9)
            ax1.annotate('PX4 End', xy=(px4_end[0], px4_end[1]), xytext=(5, -10),
                        textcoords='offset points', color='darkgreen', fontsize=9)

        # mark start / end points for VIO (raw)
        if self.vio_xyz is not None and len(self.vio_xyz) > 0:
            vio_start = self.vio_xyz[0, :2]
            vio_end = self.vio_xyz[-1, :2]
            ax1.scatter(vio_start[0], vio_start[1], c='orange', marker='o', s=80, zorder=5, label='VIO Start')
            ax1.scatter(vio_end[0], vio_end[1], c='red', marker='X', s=80, zorder=5, label='VIO End')
            ax1.annotate('VIO Start', xy=(vio_start[0], vio_start[1]), xytext=(5, 5),
                        textcoords='offset points', color='orange', fontsize=9)
            ax1.annotate('VIO End', xy=(vio_end[0], vio_end[1]), xytext=(5, -10),
                        textcoords='offset points', color='red', fontsize=9)
                        
        # mark start / end points for FUS (raw)
        if self.fus_xyz is not None and len(self.fus_xyz) > 0:
            fus_start = self.fus_xyz[0, :2]
            fus_end = self.fus_xyz[-1, :2]
            ax1.scatter(fus_start[0], fus_start[1], c='purple', marker='o', s=80, zorder=5, label='FUS Start')
            ax1.scatter(fus_end[0], fus_end[1], c='darkviolet', marker='X', s=80, zorder=5, label='FUS End')
            ax1.annotate('FUS Start', xy=(fus_start[0], fus_start[1]), xytext=(5, 5),
                        textcoords='offset points', color='purple', fontsize=9)
            ax1.annotate('FUS End', xy=(fus_end[0], fus_end[1]), xytext=(5, -10),
                        textcoords='offset points', color='darkviolet', fontsize=9)
                        
        # Compute and plot ideal trajectory point and line from PX4 start using distance & yaw inputs
        try:
            dist_text = self.distance_edit.text().strip()
            yaw_text = self.yaw_edit.text().strip()
            if dist_text and yaw_text:
                dist_val = float(dist_text)
                yaw_deg = float(yaw_text)
                yaw_rad = np.deg2rad(yaw_deg)

                # relative ideal point (East, North) based on user's formula
                ideal_rel_x = dist_val * np.cos(yaw_rad)
                ideal_rel_y = dist_val * np.sin(yaw_rad)

                # Use PX4 start as origin for the ideal trajectory if available
                if self.px4_xyz is not None and len(self.px4_xyz) > 0:
                    origin = self.px4_xyz[0, :2]
                else:
                    origin = np.array([0.0, 0.0])

                ideal_abs_x = origin[0] + ideal_rel_x
                ideal_abs_y = origin[1] + ideal_rel_y

                # Plot the ideal point and the line from start to ideal
                ax1.plot([origin[0], ideal_abs_x], [origin[1], ideal_abs_y], linestyle='--',
                        color='blue', linewidth=2, label='Ideal Trajectory')
                ax1.scatter(ideal_abs_x, ideal_abs_y, c='blue', marker='^', s=100, zorder=6, label='Ideal Point')
                ax1.annotate('Ideal Point', xy=(ideal_abs_x, ideal_abs_y), xytext=(5, 5),
                            textcoords='offset points', color='blue', fontsize=9)
        except ValueError:
            # invalid numeric input for distance/yaw — ignore ideal plotting
            pass
            
        # Configure trajectory plot
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("2D Trajectory Comparison (X-Y Plane)")
        ax1.legend()
        ax1.grid(True)
        ax1.axis("equal")
        
        # Create subplot for velocities
        ax2 = self.figure.add_subplot(212)
        
        # Get timestamps for x-axis
        timestamps = np.arange(len(self.df)) * self.period
        
        # Plot velocities if available
        # Get timestamps for x-axis
        timestamps = np.arange(len(self.df)) * self.period
        
        # Try to plot each velocity separately
        try:
            px4_cols = [f"{self.px4_vel_prefix} X", f"{self.px4_vel_prefix} Y", f"{self.px4_vel_prefix} Z"]
            if all(col in self.df.columns for col in px4_cols):
                px4_vel = np.linalg.norm(self.df[px4_cols].values, axis=1)
                ax2.plot(timestamps, px4_vel, label="PX4 Velocity", linewidth=2)
        except Exception as e:
            print(f"Could not plot PX4 velocity: {e}")

        try:
            vio_cols = [f"{self.vio_vel_prefix} X", f"{self.vio_vel_prefix} Y", f"{self.vio_vel_prefix} Z"]
            if all(col in self.df.columns for col in vio_cols):
                vio_vel = np.linalg.norm(self.df[vio_cols].values, axis=1)
                ax2.plot(timestamps, vio_vel, label="VIO Velocity", linewidth=2)
        except Exception as e:
            print(f"Could not plot VIO velocity: {e}")

        try:
            fus_cols = [f"{self.fus_vel_prefix} X", f"{self.fus_vel_prefix} Y", f"{self.fus_vel_prefix} Z"]
            if all(col in self.df.columns for col in fus_cols):
                fus_vel = np.linalg.norm(self.df[fus_cols].values, axis=1)
                ax2.plot(timestamps, fus_vel, label="FUS Velocity", linewidth=2, color='purple')
        except Exception as e:
            print(f"Could not plot FUS velocity: {e}")
            
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.set_title("Velocity Comparison")
            ax2.legend()
            ax2.grid(True)
            
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

        # Compute and plot ideal trajectory point and line from PX4 start using distance & yaw inputs
        try:
            dist_text = self.distance_edit.text().strip()
            yaw_text = self.yaw_edit.text().strip()
            if dist_text and yaw_text:
                dist_val = float(dist_text)
                yaw_deg = float(yaw_text)
                yaw_rad = np.deg2rad(yaw_deg)

                # relative ideal point (East, North) based on user's formula
                ideal_rel_x = dist_val * np.cos(yaw_rad)
                ideal_rel_y = dist_val * np.sin(yaw_rad)

                # Use PX4 start as origin for the ideal trajectory if available
                if self.px4_xyz is not None and len(self.px4_xyz) > 0:
                    origin = self.px4_xyz[0, :2]
                else:
                    origin = np.array([0.0, 0.0])

                ideal_abs_x = origin[0] + ideal_rel_x
                ideal_abs_y = origin[1] + ideal_rel_y

                # Plot the ideal point and the line from start to ideal
                ax1.plot([origin[0], ideal_abs_x], [origin[1], ideal_abs_y], linestyle='--',
                        color='blue', linewidth=2, label='Ideal Trajectory')
                ax1.scatter(ideal_abs_x, ideal_abs_y, c='blue', marker='^', s=100, zorder=6, label='Ideal Point')
                ax1.annotate('Ideal Point', xy=(ideal_abs_x, ideal_abs_y), xytext=(5, 5),
                            textcoords='offset points', color='blue', fontsize=9)
        except ValueError:
            # invalid numeric input for distance/yaw — ignore ideal plotting
            pass
        
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("2D Trajectory Comparison (X-Y Plane)")
        ax1.legend()
        ax1.grid(True)
        ax1.axis("equal")
        
        self.canvas.draw()
        
    def evaluate_trajectory(self):
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return
            
        try:
            self.load_data(file_path)
            if not self.validate_columns():
                return
                
            self.extract_trajectories()
            self.align_trajectories()
            
            self.compute_ate()
            self.compute_rpe()
            self.compute_distance_stats()
            has_velocity = self.compute_velocity_stats()
            
            # Update the actual distance field with PX4 distance
            self.actual_distance_edit.setText(f"{self.px4_total_distance:.2f}")
            
            self.plot_trajectory()
            self.export_btn.setEnabled(True)

            report_name = self.test_num_edit.text() + "_report"
            self.report_name_edit.setText(report_name)
            
            QMessageBox.information(self, "Success", "Trajectory evaluation completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            
    def generate_report_text(self):
        text = f"""
        Device Model: {self.device_model_edit.text()}
        Test Number: {self.test_num_edit.text()}
        
        == Mission Parameters ==
        Distance: {self.distance_edit.text()} m
        Yaw: {self.yaw_edit.text()} deg
        Height: {self.height_edit.text()} m
        Max Speed: {self.max_speed_edit.text()} m/s
        
        == Mission Results ==
        Actual Distance: {self.actual_distance_edit.text()} m
        Explanation: {self.result_explanation_edit.toPlainText()}
        
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

        == FUS Height Statistics ==
        Max height of  FUS estimate is {self.fus_xyz[:, 2].max():.2f}
        Min height of  FUS estimate is {self.fus_xyz[:, 2].min():.2f}
        Mean height of FUS estimate is {self.fus_xyz[:, 2].mean():.2f}
        """

        if not hasattr(self, 'missing_vel') or not self.missing_vel:
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

        == FUS Velocity Statistics ==
        Mean speed: {self.mean_fus_speed:.2f} m/s
        Max speed : {self.max_fus_speed:.2f} m/s
        Min speed : {self.min_fus_speed:.2f} m/s
        STD speed : {self.std_fus_speed:.2f} m/s
        """

        return text
        
    def export_report(self):
        if not hasattr(self, 'px4_xyz'):
            QMessageBox.warning(self, "Warning", "Please evaluate a trajectory first.")
            return
            
        # Get report name
        report_name = self.report_name_edit.text().strip()
        if not report_name:
            QMessageBox.warning(self, "Warning", "Please enter a report name.")
            return
            
        # Create output directory structure
        today_str = datetime.today().strftime("%Y_%m_%d")
        output_dir = self.data_dir / "output" / today_str / "Reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / f"{report_name}.pdf"
        
        # Create PDF report
        from matplotlib.backends.backend_pdf import PdfPages
        
        try:
            with PdfPages(str(save_path)) as pdf:
                # Plot figure
                self.figure.savefig(pdf, format='pdf', bbox_inches='tight')
                
                # Text page
                fig_text = plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.01, 0.99, self.generate_report_text(), 
                        va='top', ha='left', fontsize=10, family='monospace')
                pdf.savefig(fig_text, bbox_inches='tight')
                plt.close(fig_text)
                
            QMessageBox.information(self, "Success", f"Report saved to:\n{save_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = TrajectoryEvaluatorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
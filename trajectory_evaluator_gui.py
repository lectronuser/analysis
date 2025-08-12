import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, 
                             QGroupBox, QFormLayout, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
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
        
        # Initialize attributes
        self.df = None
        self.px4_xyz = None
        self.vio_xyz = None
        self.vio_aligned = None
        self.errors = None
        self.rpe = None
        self.output_dir = None
        self.result_output_file = None
        self.pdf_filename = None
        
        # Configuration
        self.time_column = "timestamp"
        self.px4_prefix = "PX4 Pose"
        self.vio_prefix = "VIO Pose"
        self.px4_vel_prefix = "PX4 Vel"
        self.vio_vel_prefix = "VIO Vel"
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
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pose CSV File", "", 
            "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            self.file_path_edit.setText(file_path)
            
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
            QMessageBox.critical(self, "Error", f"Missing columns in CSV: {missing}")
            return False
        return True
    
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
        
    def plot_trajectory(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(self.px4_xyz[:, 0], self.px4_xyz[:, 1], label="PX4 (Ground Truth)", linewidth=2)
        ax.plot(self.vio_aligned[:, 0], self.vio_aligned[:, 1], label="VIO (Estimated)", linewidth=2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("2D Trajectory Comparison (X-Y Plane)")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        
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
        """

        return text
        
    def export_report(self):
        if not hasattr(self, 'px4_xyz'):
            QMessageBox.warning(self, "Warning", "Please evaluate a trajectory first.")
            return
            
        # Get output directory
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", 
            "PDF Files (*.pdf);;All Files (*)", options=options)
            
        if not save_path:
            return
            
        # Create PDF report
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(save_path) as pdf:
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


def main():
    app = QApplication(sys.argv)
    window = TrajectoryEvaluatorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
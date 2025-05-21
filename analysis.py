#!/usr/bin/env python3

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D


class CSVVisualizer:
    def __init__(self, input_path=None, output_format='screen', show_dimension=None):
        self.input_path = input_path or os.path.expanduser("~/data")
        self.output_format = output_format
        self.show_dimension = show_dimension  # None, 2 or 3
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        # Define column groups
        self.column_groups = {
            '2d_pose': ("PX4 Pose X", "PX4 Pose Y", "VIO Pose X", "VIO Pose Y"),
            'distance': ("PX4 Pose Distance", "VIO Pose Distance")
        }
        
        self.movement_columns = {
            'vio': ["VIO Pose X", "VIO Pose Y", "VIO Pose Z"],
            'px4': ["PX4 Pose X", "PX4 Pose Y", "PX4 Pose Z"]
        }
        
        self.setup_output()

    def setup_output(self):
        if self.output_format in ['pdf', 'png']:
            self.output_folder = os.path.join(self.input_path, "output", self.today_str)
            os.makedirs(self.output_folder, exist_ok=True)
            
            if self.output_format == 'pdf':
                self.output_file = os.path.join(self.output_folder, f"analysis_{self.today_str}.pdf")
            else:
                self.output_file = None

    def align_trajectories(self, source, target):
        """
        Align source trajectory to target trajectory using Umeyama algorithm
        """
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
        return aligned

    def sort_natural(self, files):
        return sorted(files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

    def get_csv_files(self):
        if os.path.isfile(self.input_path):
            return [self.input_path]
        elif os.path.isdir(self.input_path):
            return self.sort_natural(glob.glob(os.path.join(self.input_path, "*.csv")))
        else:
            raise FileNotFoundError(f"Belirtilen yol bulunamadı: {self.input_path}")

    def create_plot(self, title, xlabel, ylabel):
        """Helper function to create a standardized plot"""
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt.gcf()

    def plot_graph(self, df, columns, file_name, pdf=None):
        """Generic function to plot any set of columns"""
        if not all(col in df.columns for col in columns):
            missing = [col for col in columns if col not in df.columns]
            print(f"{file_name} içinde eksik sütunlar: {', '.join(missing)}")
            return

        fig = self.create_plot(
            title=f"{file_name} - {' vs '.join(columns)}",
            xlabel="Time (s)",
            ylabel="Distance (m)"
        )
        
        info = []
        for col in columns:
            plt.plot(df[col], label=col, linestyle="-")
            
            min_val, max_val = df[col].min(), df[col].max()
            min_idx, max_idx = df[col].idxmin(), df[col].idxmax()
            avg_val = df[col].mean()
            info.append((col, min_val, max_val, avg_val))
            
            plt.scatter(min_idx, min_val, color="red", marker="o", s=50)
            plt.scatter(max_idx, max_val, color="blue", marker="o", s=50)
            
            plt.annotate(f"Min: {min_val:.2f}", (min_idx, min_val), 
                        textcoords="offset points", xytext=(-15, 10), 
                        ha='center', fontsize=8, color="red")
            plt.annotate(f"Max: {max_val:.2f}", (max_idx, max_val), 
                        textcoords="offset points", xytext=(15, -10), 
                        ha='center', fontsize=8, color="blue")
        
        plt.legend()
        self.save_or_show_plot(fig, file_name, "_".join(columns), pdf)
        
        if self.output_format == 'pdf':
            self.create_info_page(file_name, " vs ".join(columns), info, pdf)

    def plot_movement(self, df, file_name, pdf=None):
        """Plot movement in 2D or 3D based on show_dimension parameter"""
        if not self.show_dimension:
            return
            
        # Get available dimensions (2 or 3)
        dim = min(self.show_dimension, 3)  # In case someone passes > 3
        vio_cols = self.movement_columns['vio'][:dim]
        px4_cols = self.movement_columns['px4'][:dim]
        
        if not all(col in df.columns for col in vio_cols + px4_cols):
            print(f"{file_name} içinde hareket verisi için gerekli sütunlar bulunamadı.")
            return
            
        if dim == 2:
            self.plot_2d_movement(df, file_name, pdf)
        else:
            self.plot_3d_movement(df, file_name, pdf)

    def plot_2d_movement(self, df, file_name, pdf=None):
        """Plot 2D movement comparison"""
        vio_xy = df[self.movement_columns['vio'][:2]].values
        px4_xy = df[self.movement_columns['px4'][:2]].values
        vio_aligned = self.align_trajectories(vio_xy, px4_xy)
        
        fig = self.create_plot(
            title=f"{file_name} - X-Y Movement Comparison (Aligned)",
            xlabel="X Position (m)",
            ylabel="Y Position (m)"
        )
        
        # Plot trajectories
        plt.plot(px4_xy[:, 0], px4_xy[:, 1], 
                label='PX4 Movement', color='orange', linestyle='-', linewidth=2)
        plt.plot(vio_aligned[:, 0], vio_aligned[:, 1], 
                label='VIO Movement (Aligned)', color='blue', linestyle='--', linewidth=2)
        
        # Plot start/end points
        for data, color, prefix in [
            (px4_xy, 'orange', 'PX4'),
            (vio_aligned, 'blue', 'VIO')
        ]:
            plt.scatter(data[0, 0], data[0, 1], 
                       color='green', marker='o', s=100, label=f'{prefix} Start')
            plt.scatter(data[-1, 0], data[-1, 1], 
                       color='red', marker='x', s=100, label=f'{prefix} End')
        
        plt.axis('equal')
        plt.legend()
        self.save_or_show_plot(fig, f"{file_name}_2d_movement", pdf=pdf)

    def plot_3d_movement(self, df, file_name, pdf=None):
        """Plot 3D movement comparison"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vio_xyz = df[self.movement_columns['vio']].values
        px4_xyz = df[self.movement_columns['px4']].values
        vio_aligned = self.align_trajectories(vio_xyz, px4_xyz)
        
        # Plot trajectories
        ax.plot(px4_xyz[:, 0], px4_xyz[:, 1], px4_xyz[:, 2],
               label='PX4 Movement', color='orange', linestyle='-', linewidth=2)
        ax.plot(vio_aligned[:, 0], vio_aligned[:, 1], vio_aligned[:, 2],
               label='VIO Movement (Aligned)', color='blue', linestyle='--', linewidth=2)
        
        # Plot start/end points
        for data, color, prefix in [
            (px4_xyz, 'orange', 'PX4'),
            (vio_aligned, 'blue', 'VIO')
        ]:
            ax.scatter(data[0, 0], data[0, 1], data[0, 2],
                      color='green', marker='o', s=100, label=f'{prefix} Start')
            ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2],
                      color='red', marker='x', s=100, label=f'{prefix} End')
        
        ax.set_title(f"{file_name} - 3D Movement Comparison (Aligned)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.legend()
        ax.grid(True)
        
        self.save_or_show_plot(fig, f"{file_name}_3d_movement", pdf=pdf)

    def save_or_show_plot(self, fig, base_name=None, suffix="", pdf=None):
        """Handle plot output based on output format"""
        if self.output_format == 'screen':
            plt.show()
        elif self.output_format == 'pdf' and pdf:
            pdf.savefig(fig)
        elif self.output_format == 'png' and base_name:
            output_path = os.path.join(self.output_folder, f"{base_name}{suffix}.png")
            fig.savefig(output_path, dpi=300)
            print(f"PNG kaydedildi: {output_path}")
        
        plt.close(fig)

    def create_info_page(self, file_name, title, info, pdf):
        """Create information page for PDF output"""
        fig = plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.text(0.5, 1.0, f"Grafik: {file_name} - {title}", 
                fontsize=12, ha='center')
        
        cell_text = [[col, f"{min_val:.2f}", f"{max_val:.2f}", f"{avg_val:.2f}"] 
                    for col, min_val, max_val, avg_val in info]
        
        plt.table(cellText=cell_text, 
                 colLabels=["Sütun", "Min", "Max", "Ortalama"], 
                 loc="center", cellLoc='center', 
                 colWidths=[0.3, 0.2, 0.2, 0.2])
        
        pdf.savefig(fig)
        plt.close(fig)

    def process_files(self):
        csv_files = self.get_csv_files()
        
        if not csv_files:
            print("İşlenecek CSV dosyası bulunamadı.")
            return
            
        if self.output_format == 'pdf':
            with PdfPages(self.output_file) as pdf:
                for csv_file in csv_files:
                    self.process_single_file(csv_file, pdf)
            print(f"PDF oluşturuldu: {self.output_file}")
        else:
            for csv_file in csv_files:
                self.process_single_file(csv_file)

    def process_single_file(self, csv_file, pdf=None):
        try:
            df = pd.read_csv(csv_file)
            file_name = os.path.basename(csv_file).replace(".csv", "")
            
            if pdf:
                fig = plt.figure(figsize=(10, 5))
                plt.axis("off")
                plt.text(0.5, 0.5, f"Veri Dosyası: {file_name}", 
                        fontsize=14, ha='center', va='center')
                pdf.savefig(fig)
                plt.close(fig)

            # Plot all column groups
            for group_name, columns in self.column_groups.items():
                self.plot_graph(df, columns, file_name, pdf)
            
            # Plot movement if requested
            self.plot_movement(df, file_name, pdf)
                
        except Exception as e:
            print(f"{csv_file} işlenirken hata oluştu: {e}")

def main():
    parser = argparse.ArgumentParser(description='CSV veri analiz ve görselleştirme aracı')
    parser.add_argument('input', nargs='?', help='CSV dosyası veya klasör yolu (varsayılan: ~/data)')
    parser.add_argument('--out', choices=['screen', 'pdf', 'png'], default='screen',
                       help='Çıktı formatı (screen, pdf, png) - varsayılan: screen')
    parser.add_argument('--show', type=int, choices=[2, 3], 
                       help='Hareket grafiği boyutu (2: 2D, 3: 3D)')
    
    args = parser.parse_args()
    
    try:
        visualizer = CSVVisualizer(
            input_path=args.input, 
            output_format=args.out,
            show_dimension=args.show
        )
        visualizer.process_files()
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D


class CSVVisualizer:
    def __init__(self):
        self._input_path = os.path.expanduser("~/data")
        self._output_format = 'screen'
        self._show_dimension = None
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        self.column_groups = {
            '2d_pose': ("PX4 Pose X", "PX4 Pose Y", "VIO Pose X", "VIO Pose Y"),
            'distance': ("PX4 Pose Distance", "VIO Pose Distance")
        }
        
        self.movement_columns = {
            'vio': ["VIO Pose X", "VIO Pose Y", "VIO Pose Z"],
            'px4': ["PX4 Pose X", "PX4 Pose Y", "PX4 Pose Z"]
        }
        
        self._output_folder = None
        self._output_file = None

    @property
    def input_path(self):
        return self._input_path
    
    @input_path.setter
    def input_path(self, value):
        self._input_path = value
        self._validate_path()
    
    @property
    def output_format(self):
        return self._output_format
    
    @output_format.setter
    def output_format(self, value):
        if value not in ['screen', 'pdf', 'png']:
            raise ValueError("Output format must be 'screen', 'pdf' or 'png'")
        self._output_format = value
        self._setup_output()
    
    @property
    def show_dimension(self):
        return self._show_dimension
    
    @show_dimension.setter
    def show_dimension(self, value):
        if value is not None and value not in [2, 3]:
            raise ValueError("Dimension must be None, 2 or 3")
        self._show_dimension = value

    def _validate_path(self):
        if not (os.path.isfile(self._input_path) or os.path.isdir(self._input_path)):
            raise FileNotFoundError(f"Path not found: {self._input_path}")

    def _setup_output(self):
        if self._output_format in ['pdf', 'png']:
            self._output_folder = os.path.join(self._input_path, "output", self.today_str)
            os.makedirs(self._output_folder, exist_ok=True)
            
            if self._output_format == 'pdf':
                self._output_file = os.path.join(
                    self._output_folder, 
                    f"analysis_{self.today_str}.pdf"
                )
            else:
                self._output_file = None

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

    def _sort_natural(self, files):
        return sorted(files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

    def get_csv_files(self):
        if os.path.isfile(self._input_path):
            return [self._input_path]
        elif os.path.isdir(self._input_path):
            return self._sort_natural(glob.glob(os.path.join(self._input_path, "*.csv")))
        else:
            raise FileNotFoundError(f"Path not found: {self._input_path}")

    def _create_plot(self, title, xlabel, ylabel, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt.gcf()

    def plot_graph(self, df, columns, file_name, pdf=None):
        if not all(col in df.columns for col in columns):
            missing = [col for col in columns if col not in df.columns]
            print(f"Missing columns in {file_name}: {', '.join(missing)}")
            return None

        fig = self._create_plot(
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
        self._save_or_show_plot(fig, file_name, "_".join(columns), pdf)
        
        if self._output_format == 'pdf' and pdf:
            self._create_info_page(file_name, " vs ".join(columns), info, pdf)
        
        return info

    def plot_movement(self, df, file_name, pdf=None):
        if not self._show_dimension:
            return None
            
        dim = min(self._show_dimension, 3)
        vio_cols = self.movement_columns['vio'][:dim]
        px4_cols = self.movement_columns['px4'][:dim]
        
        if not all(col in df.columns for col in vio_cols + px4_cols):
            print(f"Missing movement data columns in {file_name}")
            return None
            
        if dim == 2:
            return self._plot_2d_movement(df, file_name, pdf)
        else:
            return self._plot_3d_movement(df, file_name, pdf)

    def _plot_2d_movement(self, df, file_name, pdf=None):
        vio_xy = df[self.movement_columns['vio'][:2]].values
        px4_xy = df[self.movement_columns['px4'][:2]].values
        vio_aligned, R, t = self.align_trajectories(vio_xy, px4_xy)
        
        fig = self._create_plot(
            title=f"{file_name} - X-Y Movement Comparison (Aligned)",
            xlabel="X Position (m)",
            ylabel="Y Position (m)"
        )
        
        plt.plot(px4_xy[:, 0], px4_xy[:, 1], 
                label='PX4 Movement', color='orange', linestyle='-', linewidth=2)
        plt.plot(vio_aligned[:, 0], vio_aligned[:, 1], 
                label='VIO Movement (Aligned)', color='blue', linestyle='--', linewidth=2)
        
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
        self._save_or_show_plot(fig, f"{file_name}_2d_movement", pdf=pdf)
        
        return {
            'transformation': {
                'rotation': R.tolist(),
                'translation': t.tolist()
            }
        }

    def _plot_3d_movement(self, df, file_name, pdf=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vio_xyz = df[self.movement_columns['vio']].values
        px4_xyz = df[self.movement_columns['px4']].values
        vio_aligned, R, t = self.align_trajectories(vio_xyz, px4_xyz)
        
        ax.plot(px4_xyz[:, 0], px4_xyz[:, 1], px4_xyz[:, 2],
               label='PX4 Movement', color='orange', linestyle='-', linewidth=2)
        ax.plot(vio_aligned[:, 0], vio_aligned[:, 1], vio_aligned[:, 2],
               label='VIO Movement (Aligned)', color='blue', linestyle='--', linewidth=2)
        
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
        
        self._save_or_show_plot(fig, f"{file_name}_3d_movement", pdf=pdf)
        
        return {
            'transformation': {
                'rotation': R.tolist(),
                'translation': t.tolist()
            }
        }

    def _save_or_show_plot(self, fig, base_name=None, suffix="", pdf=None):
        """Handle plot output based on output format"""
        if self._output_format == 'screen':
            plt.show()
        elif self._output_format == 'pdf' and pdf:
            pdf.savefig(fig)
        elif self._output_format == 'png' and base_name and self._output_folder:
            output_path = os.path.join(self._output_folder, f"{base_name}{suffix}.png")
            fig.savefig(output_path, dpi=300)
            print(f"Saved PNG: {output_path}")
        
        plt.close(fig)
        return output_path if self._output_format == 'png' else None

    def _create_info_page(self, file_name, title, info, pdf):
        """Create information page for PDF output"""
        fig = plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.text(0.5, 1.0, f"Graph: {file_name} - {title}", 
                fontsize=12, ha='center')
        
        cell_text = [[col, f"{min_val:.2f}", f"{max_val:.2f}", f"{avg_val:.2f}"] 
                    for col, min_val, max_val, avg_val in info]
        
        plt.table(cellText=cell_text, 
                 colLabels=["Column", "Min", "Max", "Average"], 
                 loc="center", cellLoc='center', 
                 colWidths=[0.3, 0.2, 0.2, 0.2])
        
        pdf.savefig(fig)
        plt.close(fig)

    def process_files(self):
        results = {
            'output_format': self._output_format,
            'files': [],
            'output_file': self._output_file,
            'output_folder': self._output_folder
        }
        
        try:
            csv_files = self.get_csv_files()
            
            if not csv_files:
                print("No CSV files found to process")
                return results
                
            if self._output_format == 'pdf' and self._output_file:
                with PdfPages(self._output_file) as pdf:
                    for csv_file in csv_files:
                        file_result = self._process_single_file(csv_file, pdf)
                        results['files'].append(file_result)
                print(f"PDF created: {self._output_file}")
            else:
                for csv_file in csv_files:
                    file_result = self._process_single_file(csv_file)
                    results['files'].append(file_result)
            
            return results
            
        except Exception as e:
            print(f"Error processing files: {e}")
            results['error'] = str(e)
            return results

    def _process_single_file(self, csv_file, pdf=None):
        file_result = {
            'filename': os.path.basename(csv_file),
            'plots': [],
            'movement': None,
            'error': None
        }
        
        try:
            df = pd.read_csv(csv_file)
            file_name = os.path.basename(csv_file).replace(".csv", "")
            
            if pdf:
                fig = plt.figure(figsize=(10, 5))
                plt.axis("off")
                plt.text(0.5, 0.5, f"Data File: {file_name}", 
                        fontsize=14, ha='center', va='center')
                pdf.savefig(fig)
                plt.close(fig)

            for group_name, columns in self.column_groups.items():
                plot_info = self.plot_graph(df, columns, file_name, pdf)
                if plot_info:
                    file_result['plots'].append({
                        'type': group_name,
                        'columns': columns,
                        'stats': plot_info
                    })
            
            if self._show_dimension:
                movement_result = self.plot_movement(df, file_name, pdf)
                if movement_result:
                    file_result['movement'] = movement_result
                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            file_result['error'] = str(e)
        
        return file_result
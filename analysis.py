#!/usr/bin/env python3

import os
import re
import glob
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D


class CSVVisualizer:
    def __init__(self, input_path=None, output_format='screen', show_move=False, show_move_3d=False):
        self.input_path = input_path or os.path.expanduser("~/data")
        self.output_format = output_format
        self.show_move = show_move
        self.show_move_3d = show_move_3d
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        self.column_pairs = [
            ("PX4 Pose X", "VIO Pose X"),
            ("PX4 Pose Y", "VIO Pose Y"),
            ("PX4 Pose Z", "VIO Pose Z"),
            ("PX4 Pose Distance", "VIO Pose Distance")
        ]
        
        self.vio_move = ["VIO Pose X", "VIO Pose Y", "VIO Pose Z"]
        self.px4_move = ["PX4 Pose X", "PX4 Pose Y", "PX4 Pose Z"]
        
        if output_format in ['pdf', 'png']:
            self.output_folder = os.path.join(self.input_path, "output", self.today_str)
            os.makedirs(self.output_folder, exist_ok=True)
            
            if output_format == 'pdf':
                self.output_file = os.path.join(self.output_folder, f"analysis_{self.today_str}.pdf")
            else:
                self.output_file = None

    def sort_natural(self, files):
        return sorted(files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

    def get_csv_files(self):
        if os.path.isfile(self.input_path):
            return [self.input_path]
        elif os.path.isdir(self.input_path):
            return self.sort_natural(glob.glob(os.path.join(self.input_path, "*.csv")))
        else:
            raise FileNotFoundError(f"Belirtilen yol bulunamadı: {self.input_path}")

    def plot_graph(self, df, col_pair, file_name, pdf=None):
        plt.figure(figsize=(10, 5))
        title_name = ""
        info = []
        
        for i, col in enumerate(col_pair):
            if col in df.columns:
                plt.plot(df[col], label=col, linestyle="-")
                min_index = df[col].idxmin()
                max_index = df[col].idxmax()
                min_value = df[col].min()
                max_value = df[col].max()
                avg_value = df[col].mean()
                info.append((col, min_value, max_value, avg_value))

                plt.scatter(min_index, min_value, color="red", marker="o", s=50)
                plt.scatter(max_index, max_value, color="blue", marker="o", s=50)

                plt.annotate(f"Min: {min_value:.2f}", (min_index, min_value), 
                            textcoords="offset points", xytext=(-15, 10), 
                            ha='center', fontsize=8, color="red")
                plt.annotate(f"Max: {max_value:.2f}", (max_index, max_value), 
                            textcoords="offset points", xytext=(15, -10), 
                            ha='center', fontsize=8, color="blue")
            else:
                print(f"{file_name} içinde '{col}' başlığı bulunamadı.")
            
            title_name += col + (" vs " if i < len(col_pair)-1 else "")

        self._set_plot_info(file_name + " - " + title_name)
        
        if self.output_format == 'screen':
            plt.show()
        elif self.output_format == 'pdf':
            pdf.savefig()
        elif self.output_format == 'png':
            output_path = os.path.join(self.output_folder, f"{file_name}_{'_'.join(col_pair)}.png")
            plt.savefig(output_path, dpi=300)
            print(f"PNG kaydedildi: {output_path}")
        
        plt.close()
        
        if self.output_format == 'pdf' and info:
            self._create_info_page(file_name, title_name, info, pdf)

    def plot_movement(self, df, file_name, pdf=None):
        if not (self.show_move or self.show_move_3d):
            return
            
        if self.show_move_3d and all(col in df.columns for col in self.vio_move + self.px4_move):
            self._plot_3d_movement(df, file_name, pdf)
        elif self.show_move and all(col in df.columns for col in self.vio_move[:2] + self.px4_move[:2]):
            self._plot_2d_movement(df, file_name, pdf)
        else:
            print(f"{file_name} içinde hareket verisi için gerekli sütunlar bulunamadı.")

    def _plot_2d_movement(self, df, file_name, pdf=None):
        plt.figure(figsize=(10, 10))
        
        if all(col in df.columns for col in self.vio_move[:2]):
            vio_x = df[self.vio_move[0]].to_numpy() 
            vio_y = df[self.vio_move[1]].to_numpy() 
            plt.plot(vio_x, vio_y, 
                    label='VIO Movement', color='blue', linestyle='-')
            plt.scatter(vio_x[0], vio_y[0], 
                    color='green', marker='o', s=100, label='VIO Start')
            plt.scatter(vio_x[-1], vio_y[-1], 
                    color='red', marker='x', s=100, label='VIO End')
        
        if all(col in df.columns for col in self.px4_move[:2]):
            px4_x = df[self.px4_move[0]].to_numpy()  
            px4_y = df[self.px4_move[1]].to_numpy()  
            plt.plot(px4_x, px4_y, 
                    label='PX4 Movement', color='orange', linestyle='--')
            plt.scatter(px4_x[0], px4_y[0], 
                    color='cyan', marker='o', s=100, label='PX4 Start')
            plt.scatter(px4_x[-1], px4_y[-1], 
                    color='purple', marker='x', s=100, label='PX4 End')
        
        plt.title(f"{file_name} - X-Y Movement Comparison")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        self._save_or_show_plot(file_name + "_2d_movement", pdf)

    def _plot_3d_movement(self, df, file_name, pdf=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if all(col in df.columns for col in self.vio_move):
            vio_x = df[self.vio_move[0]].to_numpy()
            vio_y = df[self.vio_move[1]].to_numpy()
            vio_z = df[self.vio_move[2]].to_numpy()
            ax.plot(vio_x, vio_y, vio_z, 
                   label='VIO Movement', color='blue', linestyle='-')
            ax.scatter(vio_x[0], vio_y[0], vio_z[0], 
                      color='green', marker='o', s=100, label='VIO Start')
            ax.scatter(vio_x[-1], vio_y[-1], vio_z[-1], 
                      color='red', marker='x', s=100, label='VIO End')
        
        if all(col in df.columns for col in self.px4_move):
            px4_x = df[self.px4_move[0]].to_numpy()
            px4_y = df[self.px4_move[1]].to_numpy()
            px4_z = df[self.px4_move[2]].to_numpy()
            ax.plot(px4_x, px4_y, px4_z, 
                   label='PX4 Movement', color='orange', linestyle='--')
            ax.scatter(px4_x[0], px4_y[0], px4_z[0], 
                      color='cyan', marker='o', s=100, label='PX4 Start')
            ax.scatter(px4_x[-1], px4_y[-1], px4_z[-1], 
                      color='purple', marker='x', s=100, label='PX4 End')
        
        ax.set_title(f"{file_name} - 3D Movement Comparison")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.legend()
        ax.grid(True)
        
        self._save_or_show_plot(file_name + "_3d_movement", pdf, fig)

    def _save_or_show_plot(self, file_name, pdf=None, fig=None):
        if fig is None:
            fig = plt.gcf()
            
        if self.output_format == 'screen':
            plt.show()
        elif self.output_format == 'pdf':
            pdf.savefig(fig)
        elif self.output_format == 'png':
            output_path = os.path.join(self.output_folder, f"{file_name}.png")
            fig.savefig(output_path, dpi=300)
            print(f"PNG kaydedildi: {output_path}")
        
        plt.close(fig)

    def _set_plot_info(self, title):
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True)

    def _create_info_page(self, file_name, title, info, pdf):
        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.text(0.5, 1.0, f"Grafik: {file_name} - {title}", 
                fontsize=12, ha='center')
        
        cell_text = []
        for col, min_val, max_val, avg_val in info:
            cell_text.append([col, f"{min_val:.2f}", f"{max_val:.2f}", f"{avg_val:.2f}"])

        column_labels = ["Sütun", "Min", "Max", "Ortalama"]
        table = plt.table(cellText=cell_text, colLabels=column_labels, 
                         loc="center", cellLoc='center', 
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        pdf.savefig()
        plt.close()

    def process_files(self):
        csv_files = self.get_csv_files()
        
        if not csv_files:
            print("İşlenecek CSV dosyası bulunamadı.")
            return
            
        if self.output_format == 'pdf':
            with PdfPages(self.output_file) as pdf:
                for csv_file in csv_files:
                    self._process_single_file(csv_file, pdf)
            print(f"PDF oluşturuldu: {self.output_file}")
        else:
            for csv_file in csv_files:
                self._process_single_file(csv_file)

    def _process_single_file(self, csv_file, pdf=None):
        try:
            df = pd.read_csv(csv_file)
            file_name = os.path.basename(csv_file).replace(".csv", "")
            
            if self.output_format == 'pdf':
                plt.figure(figsize=(10, 5))
                plt.axis("off")
                plt.text(0.5, 0.5, f"Veri Dosyası: {file_name}", 
                        fontsize=14, ha='center', va='center')
                pdf.savefig()
                plt.close()

            for col_pair in self.column_pairs:
                self.plot_graph(df, col_pair, file_name, pdf)
            
            self.plot_movement(df, file_name, pdf)
                
        except Exception as e:
            print(f"{csv_file} işlenirken hata oluştu: {e}")

def main():
    parser = argparse.ArgumentParser(description='CSV veri analiz ve görselleştirme aracı')
    parser.add_argument('input', nargs='?', help='CSV dosyası veya klasör yolu (varsayılan: ~/data)')
    parser.add_argument('--out', choices=['screen', 'pdf', 'png'], default='screen',
                       help='Çıktı formatı (screen, pdf, png) - varsayılan: screen')
    parser.add_argument('--show-move', action='store_true',
                       help='PX4 ve VIO X-Y hareket grafiğini göster')
    parser.add_argument('--show-move-3d', action='store_true',
                       help='PX4 ve VIO 3B hareket grafiğini göster (X-Y-Z)')
    
    args = parser.parse_args()
    
    try:
        visualizer = CSVVisualizer(input_path=args.input, 
                                 output_format=args.out,
                                 show_move=args.show_move,
                                 show_move_3d=args.show_move_3d)
        visualizer.process_files()
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Plotter:
    def __init__(self, enable_plt=False):
        self.enable_plt = enable_plt
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        self.data_folder = os.path.expanduser("~/data")
        self.output_folder = os.path.join(self.data_folder, self.today_str)
        self.pdf_output_path = os.path.join(self.output_folder, f"{self.today_str}.pdf")

        self.column_pairs = [
            ("PX4 Pose X", "VIO Pose X"),
            ("PX4 Pose Y", "VIO Pose Y"),
            ("PX4 Pose Z", "VIO Pose Z", "Sensor Distance")
            # ("PX4 Pose Distance", "VIO Pose Distance"),
            # ("Sensor Distance", "VIO Pose Distance"),
            # ("PX4 Pose Distance", "VIO Pose Distance"),
            # ("PX4 Pose Distance", "VIO Pose Distance"),
        ]

        self.create_folders()

    def create_folders(self):
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

    def sort_natural(self, files):
        return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    def plot_graph(self, df, col_pair, file_name, pdf):
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
                plt.annotate(f"Min: {min_value:.2f}", (min_index, min_value), textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=8, color="red")
                plt.annotate(f"Max: {max_value:.2f}", (max_index, max_value), textcoords="offset points", xytext=(15, -10), ha='center', fontsize=8, color="blue")
            
            else:
                print(f"{file_name} içinde '{col}' başlığı bulunamadı.")

            if i < len(col_pair) - 1:
                title_name += col + " vs "
            else:
                title_name += col

        self.set_plt_info(file_name)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.text(0.5, 1.0, f"Grafik: {file_name} - {title_name}", fontsize=12, ha='center')
        cell_text = []
        for col, min_val, max_val, avg_val in info:
            cell_text.append([col, f"{min_val:.2f}", f"{max_val:.2f}", f"{avg_val:.2f}"])

        column_labels = ["Grafik Adı", "Min Değer", "Max Değer", "Ortalama Değer"]
        table = plt.table(cellText=cell_text, colLabels=column_labels, loc="center", cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        pdf.savefig()
        plt.close()

            #     self.set_plt_info(f"{file_name} - {col1} vs {col2}")
            #     pdf.savefig()
            #     plt.close()

            #     # Yeni bir sayfada dosya adıyla başla
            #     plt.figure(figsize=(10, 5))
            #     plt.axis("off")
            #     plt.text(0.5, 1.0, f"Grafik: {file_name} - {col1} vs {col2}", fontsize=12, ha='center')

            #     cell_text = [
            #         [col1, f"{min_value1:.2f}", f"{max_value1:.2f}", f"{avg_value1:.2f}"],
            #         [col2, f"{min_value2:.2f}", f"{max_value2:.2f}", f"{avg_value2:.2f}"]
            #     ]
            #     column_labels = ["Veri Adı", "Min Değer", "Max Değer", "Ortalama Değer"]
            #     table = plt.table(cellText=cell_text, colLabels=column_labels, loc="center", cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
            #     table.auto_set_font_size(False)
            #     table.set_fontsize(10)
            #     table.scale(1.2, 1.2)

            #     pdf.savefig()
            #     plt.close()
            # else:
            #     print(f"{file_name} içinde '{col1}' veya '{col2}' başlığı bulunamadı.")

    def process_csv_files(self):
        csv_files = self.sort_natural(glob.glob(os.path.join(self.data_folder, "*.csv")))
        with PdfPages(self.pdf_output_path) as pdf:
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    file_name = os.path.basename(csv_file).replace(".csv", "")
                    
                    # Dosya adıyla yeni bir sayfa başlat
                    plt.figure(figsize=(10, 5))
                    plt.axis("off")
                    plt.text(0.5, 0.5, f"Veri Dosyası: {file_name}", fontsize=14, ha='center', va='center')

                    pdf.savefig()
                    plt.close()

                    for col_pair in self.column_pairs:
                        self.plot_graph(df, col_pair, file_name, pdf)

                except Exception as e:
                    print(f"{csv_file} işlenirken hata oluştu: {e}")

    def set_plt_info(self, title):
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True)   

def main():
    plotter = Plotter()
    plotter.process_csv_files()

if __name__ == "__main__":
    main()

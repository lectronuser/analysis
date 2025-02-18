#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.data_folder = os.path.expanduser("~/data")
        self.column_pairs = [
            ("PX4 Pose X", "VIO Pose X"),
            ("PX4 Pose Y", "VIO Pose Y"),
            ("PX4 Pose Z", "VIO Pose Z", "Sensor Distance")
            # ("PX4 Pose Distance", "VIO Pose Distance")
        ]
        self.create_folders()

    def create_folders(self):
        os.makedirs(self.data_folder, exist_ok=True)

    def sort_natural(self, files):
        return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    def plot_graph(self, df, col_pair, file_name):
        plt.figure(figsize=(10, 5))
        title_name = ""
        for i, col in enumerate(col_pair):
            if col in df.columns:
                plt.plot(df[col], label=col, linestyle="-")
                min_index = df[col].idxmin()
                max_index = df[col].idxmax()
                min_value = df[col].min()
                max_value = df[col].max()

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

        self.set_plt_info(file_name + " - " + title_name)
        plt.show()
        plt.close()

    def process_csv_files(self):
        csv_files = self.sort_natural(glob.glob(os.path.join(self.data_folder, "*.csv")))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_name = os.path.basename(csv_file).replace(".csv", "")

                for col_pair in self.column_pairs:
                    self.plot_graph(df, col_pair, file_name)

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

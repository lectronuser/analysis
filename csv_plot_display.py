#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, enable_plt=False):
        self.data_folder = os.path.expanduser("~/data")
        self.columns_to_plot = ["PX4 Pose Y", "VIO Pose Y"]
        self.create_folders()

    def create_folders(self):
        os.makedirs(self.data_folder, exist_ok=True)

    def sort_natural(self, files):
        return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    def plot_single_graph_image(self, df, file_name):
        plt.figure(figsize=(10, 5))
        for col in self.columns_to_plot:
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

        self.set_plt_info(file_name)
        plt.show()
        plt.close()

    def plot_multiple_subplots_image(self, df, file_name):
        fig, axes = plt.subplots(1, len(self.columns_to_plot), figsize=(15, 5))
        for ax, col in zip(axes, self.columns_to_plot):
            if col in df.columns:
                ax.plot(df[col], label=col, linestyle="-")
                min_index = df[col].idxmin()
                max_index = df[col].idxmax()
                min_value = df[col].min()
                max_value = df[col].max()

                ax.scatter(min_index, min_value, color="red", marker="o", s=50)
                ax.scatter(max_index, max_value, color="blue", marker="o", s=50)

                ax.annotate(f"Min: {min_value:.2f}", (min_index, min_value), textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=8, color="red")
                ax.annotate(f"Max: {max_value:.2f}", (max_index, max_value), textcoords="offset points", xytext=(15, -10), ha='center', fontsize=8, color="blue")

                ax.set_title(col)
                ax.set_xlabel("Time")
                ax.set_ylabel("Distance (m)")
                ax.grid(True)
            else:
                ax.set_title(f"{col} (Veri yok)")
                ax.grid(True)

        plt.tight_layout()
        plt.show()
        plt.close()

    def process_csv_files(self):
        csv_files = self.sort_natural(glob.glob(os.path.join(self.data_folder, "*.csv")))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_name = os.path.basename(csv_file).replace(".csv", "")
               
                self.plot_single_graph_image(df, file_name)
                # self.plot_multiple_subplots_image(df, file_name)

            except Exception as e:
                print(f"{csv_file} işlenirken hata oluştu: {e}")
    
    def set_plt_info(self, file_name):
        plt.title(f"{file_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True)
        
def main():
    plotter = Plotter()
    plotter.process_csv_files()

if __name__ == "__main__":
    main()
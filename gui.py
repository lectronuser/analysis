#!/usr/bin/env python3

import os
import re
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CSVVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Visualizer")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.input_path = tk.StringVar(value=os.path.expanduser("~/data"))
        self.output_format = tk.StringVar(value="screen")
        self.show_move = tk.BooleanVar(value=False)
        self.show_move_3d = tk.BooleanVar(value=False)
        self.selected_files = []
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize visualizer (without processing)
        self.visualizer = None
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input selection
        input_frame = ttk.LabelFrame(main_frame, text="Input Selection", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Input Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        ttk.Button(input_frame, text="Select Files", command=self.select_files).grid(row=1, column=2, pady=5)
        
        # Selected files listbox
        self.files_listbox = tk.Listbox(input_frame, height=5, selectmode=tk.EXTENDED)
        self.files_listbox.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Output options
        options_frame = ttk.LabelFrame(main_frame, text="Visualization Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="Output Format:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="Screen", variable=self.output_format, value="screen").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="PDF", variable=self.output_format, value="pdf").grid(row=0, column=2, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="PNG", variable=self.output_format, value="png").grid(row=0, column=3, sticky=tk.W)
        
        ttk.Checkbutton(options_frame, text="Show 2D Movement", variable=self.show_move).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Show 3D Movement", variable=self.show_move_3d).grid(row=1, column=2, sticky=tk.W, pady=2)
        
        # Graph selection
        graph_frame = ttk.LabelFrame(main_frame, text="Graph Selection", padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Available graphs (same as in CSVVisualizer class)
        self.graph_vars = {
            "PX4 vs VIO X-Y Positions": tk.BooleanVar(value=True),
            "PX4 vs VIO Distance": tk.BooleanVar(value=True),
            "VIO Position Variance": tk.BooleanVar(value=False)
        }
        
        for i, (graph_name, var) in enumerate(self.graph_vars.items()):
            ttk.Checkbutton(graph_frame, text=graph_name, variable=var).grid(row=i, column=0, sticky=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
        # Preview area
        self.preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = None
        
    def browse_input(self):
        path = filedialog.askdirectory(initialdir=self.input_path.get())
        if path:
            self.input_path.set(path)
            self.selected_files = []
            self.update_files_listbox()
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            initialdir=self.input_path.get(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.update_files_listbox()
    
    def update_files_listbox(self):
        self.files_listbox.delete(0, tk.END)
        if self.selected_files:
            for file in self.selected_files:
                self.files_listbox.insert(tk.END, os.path.basename(file))
        else:
            # Show files from directory
            path = self.input_path.get()
            if os.path.isdir(path):
                files = glob.glob(os.path.join(path, "*.csv"))
                for file in sorted(files):
                    self.files_listbox.insert(tk.END, os.path.basename(file))
    
    def clear_selection(self):
        self.selected_files = []
        self.update_files_listbox()
    
    def visualize(self):
        try:
            # Determine which graphs to show based on checkboxes
            column_pairs = []
            if self.graph_vars["PX4 vs VIO X-Y Positions"].get():
                column_pairs.append(("PX4 Pose X", "PX4 Pose Y", "VIO Pose X", "VIO Pose Y"))
            if self.graph_vars["PX4 vs VIO Distance"].get():
                column_pairs.append(("PX4 Pose Distance", "VIO Pose Distance"))
            if self.graph_vars["VIO Position Variance"].get():
                column_pairs.append(("VIO Pose Var X", "VIO Pose Var Y", "VIO Pose Var Z"))
            
            # Initialize visualizer with selected options
            self.visualizer = CSVVisualizer(
                input_path=self.input_path.get(),
                output_format=self.output_format.get(),
                show_move=self.show_move.get(),
                show_move_3d=self.show_move_3d.get()
            )
            
            # Override the column pairs with our selections
            self.visualizer.column_pairs = column_pairs
            
            # If specific files are selected, use only those
            if self.selected_files:
                self.visualizer.get_csv_files = lambda: self.selected_files
            
            # Process the files
            if self.output_format.get() == "screen":
                self.process_with_preview()
            else:
                self.visualizer.process_files()
                messagebox.showinfo("Success", f"Visualization saved to {self.visualizer.output_folder}")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def process_with_preview(self):
        # Clear previous preview
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Process first file for preview
        files = self.selected_files if self.selected_files else self.visualizer.get_csv_files()
        if not files:
            messagebox.showwarning("Warning", "No CSV files found to process")
            return
        
        try:
            df = pd.read_csv(files[0])
            file_name = os.path.basename(files[0]).replace(".csv", "")
            
            # Create preview of first graph type
            if self.visualizer.column_pairs:
                fig = plt.figure(figsize=(10, 5))
                col_pair = self.visualizer.column_pairs[0]
                
                for i, col in enumerate(col_pair):
                    if col in df.columns:
                        plt.plot(df[col], label=col, linestyle="-")
                
                self.visualizer._set_plot_info(file_name + " - " + " vs ".join(col_pair))
                
                # Embed the plot in Tkinter
                self.canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Process all files in background
                self.root.after(100, self.visualizer.process_files)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create preview: {str(e)}")


# Keep the original CSVVisualizer class unchanged
class CSVVisualizer:
    def __init__(self, input_path=None, output_format='screen', show_move=False, show_move_3d=False):
        self.input_path = input_path or os.path.expanduser("~/data")
        self.output_format = output_format
        self.show_move = show_move
        self.show_move_3d = show_move_3d
        self.today_str = datetime.today().strftime("%Y_%m_%d")
        
        self.column_pairs = [
            ("PX4 Pose X", "PX4 Pose Y", "VIO Pose X", "VIO Pose Y"),
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


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVVisualizerGUI(root)
    root.mainloop()
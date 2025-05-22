#!/usr/bin/env python3
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog,
                             QMessageBox, QProgressBar, QTabWidget, QGroupBox, QListWidget, QTextEdit)
from PyQt5.QtCore import Qt
from analysis import CSVVisualizer
from trajectory_evaluator import TrajectoryEvaluator

class AnalysisApp:
    def __init__(self):
        self.app = QApplication([])
        self.window = QMainWindow()
        self.visualizer = CSVVisualizer()
        self.evaluator = TrajectoryEvaluator()
        self.setup_ui()
        self.current_mode = "visualization"  

    def setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        vis_tab = QWidget()
        self.setup_visualization_tab(vis_tab)
        self.tabs.addTab(vis_tab, "CSV Görselleştirme")
        
        eval_tab = QWidget()
        self.setup_evaluation_tab(eval_tab)
        self.tabs.addTab(eval_tab, "Trajectory Değerlendirme")
        
        layout.addWidget(self.tabs)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        central_widget.setLayout(layout)
        self.window.setCentralWidget(central_widget)
        self.window.setWindowTitle("Veri Analiz Aracı")
        # self.window.resize(720, 480)
        # self.window.showFullScreen() 
        self.window.showMaximized()

    def setup_visualization_tab(self, tab):
        layout = QVBoxLayout()
        
        file_group = QGroupBox("Veri Kaynağı")
        file_layout = QHBoxLayout()
        
        file_layout.addWidget(QLabel("CSV Dosya/Klasör:"))
        self.vis_file_input = QLineEdit(os.path.expanduser("~/data"))
        file_layout.addWidget(self.vis_file_input)
        
        browse_btn = QPushButton("Gözat...")
        browse_btn.clicked.connect(lambda: self.browse_files(self.vis_file_input))
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        self.file_list_group = QGroupBox("Seçilecek Dosyalar")
        file_list_layout = QVBoxLayout()
        
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QListWidget.MultiSelection)
        file_list_layout.addWidget(self.file_list_widget)
        
        refresh_btn = QPushButton("Listeyi Yenile")
        refresh_btn.clicked.connect(self.refresh_file_list)
        file_list_layout.addWidget(refresh_btn)
        
        self.file_list_group.setLayout(file_list_layout)
        layout.addWidget(self.file_list_group)
        
        param_group = QGroupBox("Görselleştirme Parametreleri")
        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel("Çıktı Formatı:"))
        self.vis_format_combo = QComboBox()
        self.vis_format_combo.addItems(["screen", "pdf", "png"])
        param_layout.addWidget(self.vis_format_combo)
        
        param_layout.addWidget(QLabel("Hareket Görselleştirme:"))
        self.vis_dim_combo = QComboBox()
        self.vis_dim_combo.addItems(["None", "2D", "3D"])
        param_layout.addWidget(self.vis_dim_combo)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        vis_run_btn = QPushButton("Görselleştirmeyi Başlat")
        vis_run_btn.clicked.connect(self.run_visualization)
        layout.addWidget(vis_run_btn)
        
        tab.setLayout(layout)
        self.refresh_file_list()  

    def refresh_file_list(self):
        self.file_list_widget.clear()
        path = self.vis_file_input.text()
        
        if os.path.isdir(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            for file in sorted(csv_files):
                self.file_list_widget.addItem(file)
        elif os.path.isfile(path) and path.endswith('.csv'):
            self.file_list_widget.addItem(os.path.basename(path))

    def setup_evaluation_tab(self, tab):
        layout = QVBoxLayout()
    
        file_group = QGroupBox("Veri Kaynağı")
        file_layout = QHBoxLayout()
        
        file_layout.addWidget(QLabel("Veri Klasörü:"))
        self.eval_file_input = QLineEdit(os.path.expanduser("~/data"))
        file_layout.addWidget(self.eval_file_input)
        
        browse_btn = QPushButton("Gözat...")
        browse_btn.clicked.connect(lambda: self.browse_files(self.eval_file_input, is_dir=True))
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        param_group = QGroupBox("Değerlendirme Parametreleri")
        param_layout = QVBoxLayout()
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Dosya Numarası:"))
        self.eval_file_index = QLineEdit("1")
        h_layout.addWidget(self.eval_file_index)
        param_layout.addLayout(h_layout)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Örnekleme Periyodu (s):"))
        self.eval_period = QLineEdit("0.1")
        h_layout.addWidget(self.eval_period)
        param_layout.addLayout(h_layout)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("RPE Delta (örnek sayısı):"))
        self.eval_rpe_delta = QLineEdit("1")
        h_layout.addWidget(self.eval_rpe_delta)
        param_layout.addLayout(h_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFontFamily("Courier") 
        layout.addWidget(QLabel("Değerlendirme Sonuçları:"))
        layout.addWidget(self.results_text)
        
        eval_run_btn = QPushButton("Değerlendirmeyi Başlat")
        eval_run_btn.clicked.connect(self.run_evaluation)
        layout.addWidget(eval_run_btn)
        
        tab.setLayout(layout)

    def browse_files(self, target_input, is_dir=False):
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self.window, 
                "Klasör Seçin", 
                os.path.expanduser("~")
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self.window,
                "Dosya Seçin",
                os.path.expanduser("~"),
                "CSV Files (*.csv);;All Files (*)"
            )
        
        if path:
            target_input.setText(path)

    def run_visualization(self):
        try:
            self.visualizer.input_path = self.vis_file_input.text()
            self.visualizer.output_format = self.vis_format_combo.currentText()
            
            dim_map = {"None": None, "2D": 2, "3D": 3}
            self.visualizer.show_dimension = dim_map[self.vis_dim_combo.currentText()]
            
            selected_files = [item.text() for item in self.file_list_widget.selectedItems()]
            if not selected_files:  
                selected_files = [self.file_list_widget.item(i).text() 
                                for i in range(self.file_list_widget.count())]
            
            self.progress.setVisible(True)
            self.progress.setRange(0, len(selected_files))
            
            results = {'files': []}
            
            for i, filename in enumerate(selected_files):
                QApplication.processEvents()
                self.progress.setValue(i + 1)
                
                full_path = os.path.join(self.visualizer.input_path, filename)
                file_result = self.visualizer._process_single_file(full_path)
                results['files'].append(file_result)
            
            self.progress.setVisible(False)
            
        except Exception as e:
            QMessageBox.critical(
                self.window, 
                "Hata", 
                f"Görselleştirme hatası:\n{str(e)}"
            )
        finally:
            self.progress.setVisible(False)

    def run_evaluation(self):
        try:
            self.evaluator.input_path = self.eval_file_input.text()
            self.evaluator.file_index = self.eval_file_index.text()
            self.evaluator.period = float(self.eval_period.text())
            self.evaluator.rpe_delta = int(self.eval_rpe_delta.text())
            
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            QApplication.processEvents()
            
            result = self.evaluator.run_evaluation()

            if result['success']:
                df = self.evaluator.load_data()
                results = self.evaluator.evaluate_trajectory(df)
                report_text = self.evaluator._generate_report_text(results)
                self.results_text.setPlainText(report_text)
            
            self.progress.setVisible(False)
            
            msg = QMessageBox()
            if result['success']:
                msg.setIcon(QMessageBox.Information)
                msg.setText("Değerlendirme Tamamlandı!")
                msg.setInformativeText(f"Rapor oluşturuldu:\n{result['report_file']}")
            else:
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Değerlendirme Hatası!")
                msg.setInformativeText(result['error'])
            
            msg.exec_()
            
        except ValueError as e:
            QMessageBox.critical(
                self.window,
                "Geçersiz Parametre",
                f"Lütfen geçerli sayısal değerler girin:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self.window,
                "Hata",
                f"Değerlendirme hatası:\n{str(e)}"
            )
        finally:
            self.progress.setVisible(False)

    def run(self):
        self.window.show()
        self.app.exec_()

if __name__ == "__main__":
    app = AnalysisApp()
    app.run()
#!/usr/bin/env python3
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog,
                             QMessageBox, QProgressBar, QTabWidget, QGroupBox)
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
        self.current_mode = "visualization"  # or "evaluation"

    def setup_ui(self):
        """Arayüzü ve sekmeleri oluşturur"""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Sekmeler
        self.tabs = QTabWidget()
        
        # 1. Görselleştirme Sekmesi
        vis_tab = QWidget()
        self.setup_visualization_tab(vis_tab)
        self.tabs.addTab(vis_tab, "CSV Görselleştirme")
        
        # 2. Trajectory Değerlendirme Sekmesi
        eval_tab = QWidget()
        self.setup_evaluation_tab(eval_tab)
        self.tabs.addTab(eval_tab, "Trajectory Değerlendirme")
        
        layout.addWidget(self.tabs)
        
        # Ortak İlerleme Çubuğu
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        central_widget.setLayout(layout)
        self.window.setCentralWidget(central_widget)
        self.window.setWindowTitle("Veri Analiz Aracı")
        self.window.resize(700, 400)

    def setup_visualization_tab(self, tab):
        """Görselleştirme sekmesi arayüzü"""
        layout = QVBoxLayout()
        
        # Dosya Seçim
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
        
        # Parametreler
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
        
        # Buton
        vis_run_btn = QPushButton("Görselleştirmeyi Başlat")
        vis_run_btn.clicked.connect(self.run_visualization)
        layout.addWidget(vis_run_btn)
        
        tab.setLayout(layout)

    def setup_evaluation_tab(self, tab):
        """Değerlendirme sekmesi arayüzü"""
        layout = QVBoxLayout()
        
        # Dosya Seçim
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
        
        # Parametreler
        param_group = QGroupBox("Değerlendirme Parametreleri")
        param_layout = QVBoxLayout()
        
        # File index
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Dosya Numarası:"))
        self.eval_file_index = QLineEdit("1")
        h_layout.addWidget(self.eval_file_index)
        param_layout.addLayout(h_layout)
        
        # Sample period
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Örnekleme Periyodu (s):"))
        self.eval_period = QLineEdit("0.1")
        h_layout.addWidget(self.eval_period)
        param_layout.addLayout(h_layout)
        
        # RPE delta
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("RPE Delta (örnek sayısı):"))
        self.eval_rpe_delta = QLineEdit("1")
        h_layout.addWidget(self.eval_rpe_delta)
        param_layout.addLayout(h_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Buton
        eval_run_btn = QPushButton("Değerlendirmeyi Başlat")
        eval_run_btn.clicked.connect(self.run_evaluation)
        layout.addWidget(eval_run_btn)
        
        tab.setLayout(layout)

    def browse_files(self, target_input, is_dir=False):
        """Dosya veya klasör seçim diyaloğu"""
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
        """Görselleştirme işlemini başlatır"""
        try:
            # Parametreleri ayarla
            self.visualizer.input_path = self.vis_file_input.text()
            self.visualizer.output_format = self.vis_format_combo.currentText()
            
            dim_map = {"None": None, "2D": 2, "3D": 3}
            self.visualizer.show_dimension = dim_map[self.vis_dim_combo.currentText()]
            
            # İlerleme çubuğunu göster
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # Belirsiz mod
            
            # İşlemi başlat
            QApplication.processEvents()
            results = self.visualizer.process_files()
            
            # Sonuçları göster
            self.progress.setVisible(False)
            
            msg = QMessageBox()
            if 'error' in results:
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Hata oluştu!")
                msg.setInformativeText(results['error'])
            else:
                msg.setIcon(QMessageBox.Information)
                output_loc = results.get('output_file', 'ekranda gösterildi')
                msg.setText(f"Görselleştirme tamamlandı!\nÇıktı: {output_loc}")
            
            msg.exec_()
            
        except Exception as e:
            QMessageBox.critical(
                self.window, 
                "Hata", 
                f"Görselleştirme hatası:\n{str(e)}"
            )
        finally:
            self.progress.setVisible(False)

    def run_evaluation(self):
        """Trajectory değerlendirme işlemini başlatır"""
        try:
            # Parametreleri ayarla
            self.evaluator.input_path = self.eval_file_input.text()
            self.evaluator.file_index = self.eval_file_index.text()
            self.evaluator.period = float(self.eval_period.text())
            self.evaluator.rpe_delta = int(self.eval_rpe_delta.text())
            
            # İlerleme çubuğunu göster
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            QApplication.processEvents()
            
            # İşlemi başlat
            result = self.evaluator.run_evaluation()
            
            # Sonuçları göster
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
        """Uygulamayı başlatır"""
        self.window.show()
        self.app.exec_()

if __name__ == "__main__":
    app = AnalysisApp()
    app.run()
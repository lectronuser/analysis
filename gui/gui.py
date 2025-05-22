#!/usr/bin/env python3
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFileDialog, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt
from analysis import CSVVisualizer

class AnalysisApp:
    def __init__(self):
        self.app = QApplication([])
        self.window = QMainWindow()
        self.visualizer = CSVVisualizer()
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("CSV Dosya/Klasör:"))
        
        self.file_input = QLineEdit(os.path.expanduser("~/data"))
        file_layout.addWidget(self.file_input)
        
        browse_btn = QPushButton("Gözat...")
        browse_btn.clicked.connect(self.browse_files)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)

        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel("Çıktı Formatı:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["screen", "pdf", "png"])
        param_layout.addWidget(self.format_combo)
        
        param_layout.addWidget(QLabel("Hareket Görselleştirme:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["None", "2D", "3D"])
        param_layout.addWidget(self.dim_combo)
        
        layout.addLayout(param_layout)

        btn_layout = QHBoxLayout()
        
        run_btn = QPushButton("Analizi Başlat")
        run_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(run_btn)
        
        cancel_btn = QPushButton("İptal")
        cancel_btn.clicked.connect(self.window.close)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        central_widget.setLayout(layout)
        self.window.setCentralWidget(central_widget)
        self.window.setWindowTitle("CSV Veri Analiz Aracı")
        self.window.resize(600, 200)

    def browse_files(self):
        path = QFileDialog.getExistingDirectory(
            self.window, 
            "Klasör Seçin", 
            os.path.expanduser("~")
        )
        if path:
            self.file_input.setText(path)

    def run_analysis(self):
        try:
            self.visualizer.input_path = self.file_input.text()
            self.visualizer.output_format = self.format_combo.currentText()
            
            dim_map = {"None": None, "2D": 2, "3D": 3}
            self.visualizer.show_dimension = dim_map[self.dim_combo.currentText()]
            
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  
            
            QApplication.processEvents()
            results = self.visualizer.process_files()
            
            self.progress.setVisible(False)
            
            msg = QMessageBox()
            if 'error' in results:
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Hata oluştu!")
                msg.setInformativeText(results['error'])
            else:
                msg.setIcon(QMessageBox.Information)
                output_loc = results.get('output_file', 'ekranda gösterildi')
                msg.setText(f"Analiz tamamlandı!\nÇıktı: {output_loc}")
            
            msg.exec_()
            
        except Exception as e:
            QMessageBox.critical(
                self.window, 
                "Hata", 
                f"Bir hata oluştu:\n{str(e)}"
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
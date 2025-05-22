#!/usr/bin/env python3
import sys
from analysis import CSVVisualizer
from gui import AnalysisApp

def main():
    if len(sys.argv) == 1:
        app = AnalysisApp()
        app.run()
    else:
        run_cli()

def run_cli():
    """Konsol arayüzü için temel fonksiyon"""
    print("CSV Analiz Aracı - Konsol Modu")
    visualizer = CSVVisualizer()
    
    visualizer.input_path = input("CSV dosya yolu veya klasörü: ").strip()
    visualizer.output_format = input("Çıktı formatı [screen/pdf/png]: ").strip()
    
    dim = input("Hareket görselleştirme boyutu [None/2/3]: ").strip()
    visualizer.show_dimension = int(dim) if dim and dim.isdigit() else None
    
    results = visualizer.process_files()
    print(f"İşlem tamamlandı! Sonuçlar: {results.get('output_file', 'Ekranda gösterildi')}")

if __name__ == "__main__":
    main()
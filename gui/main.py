#!/usr/bin/env python3
import sys
from analysis import CSVVisualizer
from trajectory_evaluator import TrajectoryEvaluator
from gui import AnalysisApp

def main():
    if len(sys.argv) == 1:
        # GUI modu
        app = AnalysisApp()
        app.run()
    else:
        # Konsol modu
        run_cli()

def run_cli():
    print("Trajectory Analysis Tool - CLI Mode")
    print("1. CSV Visualization")
    print("2. Trajectory Evaluation")
    choice = input("Select mode (1/2): ").strip()
    
    if choice == "1":
        visualizer = CSVVisualizer()
        visualizer.input_path = input("CSV file/directory path: ").strip()
        visualizer.output_format = input("Output format [screen/pdf/png]: ").strip()
        dim = input("Movement visualization [None/2/3]: ").strip()
        visualizer.show_dimension = int(dim) if dim and dim.isdigit() else None
        visualizer.process_files()
    elif choice == "2":
        evaluator = TrajectoryEvaluator()
        evaluator.input_path = input("Data directory path: ").strip()
        evaluator.file_index = input("File index (e.g., 1 for pose_1.csv): ").strip()
        result = evaluator.run_evaluation()
        
        if result['success']:
            print(f"Report generated: {result['report_file']}")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
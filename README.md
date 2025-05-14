# analysis

```bash
./analysis.py --help
usage: analysis.py [-h] [--out {screen,pdf,png}] [--show-move] [--show-move-3d] [input]

CSV veri analiz ve görselleştirme aracı

positional arguments:
  input                 CSV dosyası veya klasör yolu (varsayılan: ~/data)

options:
  -h, --help            show this help message and exit
  --out {screen,pdf,png}
                        Çıktı formatı (screen, pdf, png) - varsayılan: screen
  --show-move           PX4 ve VIO X-Y hareket grafiğini göster
  --show-move-3d        PX4 ve VIO 3B hareket grafiğini göster (X-Y-Z)

```

## trajectory_evaluator.py
This file evaluate the one single cvs file under data directory that is existed in your home directory.
And the name of files must be "pose_n.csv". n indicates the number of the file.
#### Usage
```bash
./trajectory_evaluator.py [number of the file]

# example
./trajectory_evaluator.py 4
```
this command will make trajectory_evaluator.py to read /home/your_user/data/pose_3.csv file and create new
file named pose_3_report.pdf under the data directory and write results to this file.

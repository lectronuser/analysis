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
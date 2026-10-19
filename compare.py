import csv
import json
from pathlib import Path

RUNS_DIR = Path("runs")
COLS = [  
    ("name", "Эксперимент", "{}"),
    ("trainable_params_M", "Обуч. парам., M", "{:.3f}"),
    ("best_miou", "mIoU (лучший)", "{:.4f}"),
    ("best_epoch", "Эпоха", "{}"),
    ("final_miou", "mIoU (последн.)", "{:.4f}"),
    ("mean_epoch_time_s", "Время эпохи, с", "{:.1f}"),
    ("train_time_min", "Всего, мин", "{:.1f}"),
    ("train_peak_mem_MB", "Память обуч., МБ", "{:.0f}"),
    ("params_M", "Параметры, M", "{:.2f}"),
    ("gmacs", "GMACs", "{:.1f}"),
    ("latency_fp32_ms", "Задержка fp32, мс", "{:.1f}"),
    ("latency_fp16_ms", "Задержка fp16, мс", "{:.1f}"),
    ("infer_mem_fp32_MB", "Память инфер., МБ", "{:.0f}"),
]


def fmt(run, key, f):
    return f.format(run[key]) if run.get(key) is not None else "—"


def main():
    runs = []
    for p in sorted(RUNS_DIR.glob("*/summary.json")):
        r = json.loads(p.read_text())
        r["name"] = p.parent.name          
        runs.append(r)
    if not runs:
        print(f"нет {RUNS_DIR}/*/summary.json")
        return

    print("| " + " | ".join(c[1] for c in COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in runs:
        print("| " + " | ".join(fmt(r, k, f) for k, _, f in COLS) + " |")

    out = RUNS_DIR / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in COLS])
        for r in runs:
            w.writerow([r.get(k) for k, _, _ in COLS])
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()

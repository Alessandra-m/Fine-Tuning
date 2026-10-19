import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
FIG_PATH = Path("reports/figures/curves.png")
DEFAULT_RUNS = ["lastlayer_ce_e50", "head_scratch_ce_e50", "full_ce_e50", "full_ce_lovasz_e50"]


def read_history(name):
    with open(RUNS_DIR / name / "history.csv") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    miou = [float(r["val_miou"]) for r in rows]
    return ep, loss, miou


def main(names):
    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        ep, loss, miou = read_history(name)
        ax_loss.plot(ep, loss, label=name)
        ax_miou.plot(ep, miou, label=name)
    ax_loss.set_title("Train loss")
    ax_loss.set_xlabel("эпоха")
    ax_miou.set_title("mIoU на валидации")
    ax_miou.set_xlabel("эпоха")
    ax_miou.set_ylim(0, 0.85)
    for ax in (ax_loss, ax_miou):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=110)
    print(f"Сохранено: {FIG_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_RUNS)

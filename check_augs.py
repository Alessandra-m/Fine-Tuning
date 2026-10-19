from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from camvid import (CLASSES, IGNORE_INDEX, NUM_CLASSES, PALETTE, CamVid, denorm,
                    get_loaders, get_transforms)

FIG_PATH = Path("reports/figures/augs_check.png")
N_IMAGES = 3       
N_AUGS = 3              
N_VALUE_CHECKS = 100    

PAL = np.array(PALETTE, dtype=np.uint8)


def overlay(img_u8, mask, alpha=0.5):
    color = PAL[mask.numpy()]
    return (img_u8 * (1 - alpha) + color * alpha).astype(np.uint8)


def check_mask_values(aug):
    allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    seen = set()
    for i in range(min(N_VALUE_CHECKS, len(aug))):
        img, mask = aug[i]
        assert img.shape[-2:] == mask.shape, f"{i}: формы {img.shape} и {mask.shape}"
        assert mask.dtype == torch.int64
        seen |= set(mask.unique().tolist())
    bad = seen - allowed
    assert not bad, f"в масках появились недопустимые значения: {sorted(bad)}"
    print(f"OK: значения масок после аугментаций {sorted(seen)}")


def check_batches():
    train_dl, val_dl = get_loaders(batch_size=4, num_workers=2)
    x, y = next(iter(train_dl))
    print(f"OK: train batch {tuple(x.shape)} {x.dtype}, masks {tuple(y.shape)} {y.dtype}")
    x, y = next(iter(val_dl))
    print(f"OK: val batch   {tuple(x.shape)} {x.dtype}, masks {tuple(y.shape)} {y.dtype}")
    print(f"train: {len(train_dl.dataset)}, val: {len(val_dl.dataset)} изображений")


def save_figure(raw, aug):
    idxs = torch.randperm(len(raw))[:N_IMAGES].tolist()
    fig, axes = plt.subplots(N_IMAGES, N_AUGS + 1, figsize=(4 * (N_AUGS + 1), 3.2 * N_IMAGES))
    for r, i in enumerate(idxs):
        img, mask = raw[i]
        axes[r, 0].imshow(overlay(denorm(img), mask))
        axes[r, 0].set_title(f"#{i} исходная")
        for c in range(1, N_AUGS + 1):
            img, mask = aug[i]
            axes[r, c].imshow(overlay(denorm(img), mask))
            axes[r, c].set_title(f"#{i} aug {c}")
    for ax in axes.ravel():
        ax.axis("off")
    legend = [Patch(color=PAL[k] / 255, label=name) for k, name in enumerate(CLASSES + ["void"])]
    fig.legend(handles=legend, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=90)
    print(f"Сохранено: {FIG_PATH}")


def main():
    torch.manual_seed(0)
    raw = CamVid(split="train", transforms=get_transforms(train=False))
    aug = CamVid(split="train", transforms=get_transforms(train=True))
    check_mask_values(aug)
    check_batches()
    save_figure(raw, aug)


if __name__ == "__main__":
    main()

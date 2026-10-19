import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from camvid import (CamVid, get_transforms, get_loaders, CLASSES, PALETTE,
                    MEAN, STD, NUM_CLASSES, IGNORE_INDEX)

PAL = np.array(PALETTE, dtype=np.uint8)


def denorm(img):
    img = img * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(img_u8, mask, alpha=0.5):
    color = PAL[mask.numpy()]
    return (img_u8 * (1 - alpha) + color * alpha).astype(np.uint8)


def main(n_images=3, n_augs=3, n_value_checks=100):
    torch.manual_seed(0)
    raw = CamVid(split="train", transforms=get_transforms(train=False))
    aug = CamVid(split="train", transforms=get_transforms(train=True))

    # 1. значения масок после аугментаций
    allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    seen = set()
    for i in range(min(n_value_checks, len(aug))):
        img, mask = aug[i]
        assert img.shape[-2:] == mask.shape, f"{i}: формы {img.shape} и {mask.shape}"
        assert mask.dtype == torch.int64
        seen |= set(mask.unique().tolist())
    bad = seen - allowed
    assert not bad, f"в масках появились недопустимые значения: {sorted(bad)}"
    print(f"OK: значения масок после аугментаций {sorted(seen)}")

    # 2. батч
    train_dl, val_dl = get_loaders(batch_size=4, num_workers=2)
    x, y = next(iter(train_dl))
    print(f"OK: train batch {tuple(x.shape)} {x.dtype}, masks {tuple(y.shape)} {y.dtype}")
    x, y = next(iter(val_dl))
    print(f"OK: val batch   {tuple(x.shape)} {x.dtype}, masks {tuple(y.shape)} {y.dtype}")
    print(f"train: {len(train_dl.dataset)}, val: {len(val_dl.dataset)} изображений")

    # 3. визуализация
    idxs = torch.randperm(len(raw))[:n_images].tolist()
    fig, axes = plt.subplots(n_images, n_augs + 1,
                             figsize=(4 * (n_augs + 1), 3.2 * n_images))
    for r, i in enumerate(idxs):
        img, mask = raw[i]
        axes[r, 0].imshow(overlay(denorm(img), mask))
        axes[r, 0].set_title(f"#{i} исходная")
        for c in range(1, n_augs + 1):
            img, mask = aug[i]
            axes[r, c].imshow(overlay(denorm(img), mask))
            axes[r, c].set_title(f"#{i} aug {c}")
    for ax in axes.ravel():
        ax.axis("off")
    legend = [Patch(color=PAL[k] / 255, label=name)
              for k, name in enumerate(CLASSES + ["void"])]
    fig.legend(handles=legend, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("result/augs_check.png", dpi=90)
    print("Сохранено: augs_check.png")


if __name__ == "__main__":
    main()

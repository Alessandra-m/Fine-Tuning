import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from camvid import CLASSES, IGNORE_INDEX, NUM_CLASSES, PALETTE, CamVid, denorm, get_transforms
from seg_utils import get_device, load_model, metrics_from_cm, update_cm

BAND_WIDTH = 3         
TOP_CONFUSIONS = 10

PAL = np.array(PALETTE, dtype=np.uint8)


def boundary_band(y, width=BAND_WIDTH):
    edge = torch.zeros_like(y, dtype=torch.bool)
    edge[1:, :] |= y[1:, :] != y[:-1, :]
    edge[:, 1:] |= y[:, 1:] != y[:, :-1]
    band = F.max_pool2d(edge[None, None].float(), 2 * width + 1, stride=1, padding=width)
    return band[0, 0] > 0


def image_miou(pred, y):
    valid = y != IGNORE_INDEX
    ious = []
    for c in range(NUM_CLASSES):
        p, g = (pred == c) & valid, (y == c) & valid
        union = (p | g).sum().item()
        if union:
            ious.append((p & g).sum().item() / union)
    return float(np.mean(ious)) if ious else float("nan")


@torch.no_grad()
def predict(model, x, device):
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return model(x[None].to(device)).argmax(1)[0].cpu()


def save_example(model, ds, i, score, path, device):
    x, y = ds[i]
    pred = predict(model, x, device)
    valid = y != IGNORE_INDEX
    err = np.zeros((*y.shape, 3), dtype=np.uint8)
    err[((pred != y) & valid).numpy()] = (255, 60, 60)
    err[(~valid).numpy()] = (90, 90, 90)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    panels = [denorm(x), PAL[y.numpy()], PAL[pred.numpy()], err]
    titles = ["изображение", "GT", "предсказание", "ошибки (серое — void)"]
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{ds.images[i].stem}: mIoU {score:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_confusion(cmn, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("GT")
    for g in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            v = cmn[g, p].item()
            if v >= 0.01:
                ax.text(p, g, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--worst", type=int, default=6)
    ap.add_argument("--best", type=int, default=3)
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device).eval()
    ds = CamVid(split="val", transforms=get_transforms(train=False))
    out = Path(args.ckpt).parent / "analysis"
    out.mkdir(exist_ok=True)

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    per_image, err_total, err_in_band, band_px, valid_px = [], 0, 0, 0, 0
    for i in range(len(ds)):
        x, y = ds[i]
        pred = predict(model, x, device)
        update_cm(cm, pred, y)
        per_image.append((image_miou(pred, y), i))

        valid = y != IGNORE_INDEX
        err = (pred != y) & valid
        band = boundary_band(y) & valid
        err_total += err.sum().item()
        err_in_band += (err & band).sum().item()
        band_px += band.sum().item()
        valid_px += valid.sum().item()

    m = metrics_from_cm(cm)
    print(f"mIoU {m['miou']:.4f}, pixel acc {m['pixel_acc']:.4f}\n")
    print("IoU по классам (по возрастанию):")
    for c, v in sorted(zip(CLASSES, m["iou"]), key=lambda t: t[1]):
        print(f"  {c:11s} {v:.3f}")

    cmn = cm.double() / cm.sum(1, keepdim=True).clamp(min=1)    # нормировка по строкам (GT)
    off = cmn.clone()
    off.fill_diagonal_(0)
    print("\nСамые частые путаницы (GT → предсказание, доля пикселей GT-класса):")
    for k in off.flatten().argsort(descending=True)[:TOP_CONFUSIONS].tolist():
        g, p = divmod(k, NUM_CLASSES)
        print(f"  {CLASSES[g]:11s} → {CLASSES[p]:11s} {off[g, p]:.3f}")

    print(f"\nГраницы (±{BAND_WIDTH} px): {100*band_px/valid_px:.1f}% пикселей, "
          f"но {100*err_in_band/max(err_total, 1):.1f}% всех ошибок")
    save_confusion(cmn, out / "confusion.png")

    per_image.sort()
    groups = [
        ("worst", "Худшие", per_image[:args.worst]),
        ("best", "Лучшие", per_image[::-1][:args.best]),
        ("median", "Медианный", [per_image[len(per_image) // 2]]),
    ]
    for prefix, title, items in groups:
        print(f"\n{title} кадры по mIoU:")
        for rank, (score, i) in enumerate(items, 1):
            name = ds.images[i].stem
            save_example(model, ds, i, score, out / f"{prefix}_{rank:02d}_{name}.png", device)
            print(f"  {rank}. {name}  mIoU {score:.3f}")
    print(f"\nКартинки в {out}/")


if __name__ == "__main__":
    main()

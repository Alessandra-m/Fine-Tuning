import argparse

import torch

from camvid import CLASSES, NUM_CLASSES, make_loader
from seg_utils import get_device, load_model, metrics_from_cm, update_cm


@torch.no_grad()
def run(model, dl, device, tta):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            if tta:
                logits = logits + model(x.flip(-1)).flip(-1)
        update_cm(cm, logits.argmax(1), y)
    return metrics_from_cm(cm.cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device).eval()
    dl = make_loader(args.split)

    base = run(model, dl, device, tta=False)
    flip = run(model, dl, device, tta=True)

    print(f"{args.ckpt}  [{args.split}, {len(dl.dataset)} кадров]")
    print(f"  без TTA:  mIoU {base['miou']:.4f}  pixel acc {base['pixel_acc']:.4f}")
    print(f"  flip-TTA: mIoU {flip['miou']:.4f}  pixel acc {flip['pixel_acc']:.4f}  "
          f"(Δ {flip['miou'] - base['miou']:+.4f})")
    print("\n  IoU по классам: без TTA → flip-TTA")
    for c, a, b in zip(CLASSES, base["iou"], flip["iou"]):
        print(f"    {c:11s} {a:.3f} → {b:.3f}  ({b - a:+.3f})")


if __name__ == "__main__":
    main()

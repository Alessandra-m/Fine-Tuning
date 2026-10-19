import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from bench import benchmark, count_params
from camvid import CLASSES, IGNORE_INDEX, NUM_CLASSES, CamVid, get_loaders, get_transforms
from seg_utils import build_model, evaluate, get_device, load_model, set_seed

BATCH_SIZE = 4
LR_NEW = 1e-3          
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4
USE_AMP = True
OUT_DIR = Path("runs")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["lastlayer", "head", "full"], required=True)
    ap.add_argument("--head-init", choices=["pretrained", "scratch"], default="pretrained")
    ap.add_argument("--loss", choices=["ce", "wce", "ce_lovasz"], default="ce")
    ap.add_argument("--lr-pretrained", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--init", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    assert args.head_init == "pretrained" or args.train == "head", \
        "--head-init scratch имеет смысл только с --train head"
    assert args.init is None or args.train == "full", "--init используется с --train full"
    return args


def lr_token(lr):
    """1e-4 -> '1e-4', 0.001 -> '1e-3'."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def default_name(args):
    if args.init:
        return f"{Path(args.init).parent.name}_ft_{args.loss}_e{args.epochs}"
    parts = [args.train]
    if args.train == "head":
        parts.append(args.head_init)
        if args.head_init == "pretrained":
            parts.append("lr" + lr_token(args.lr_pretrained))
    parts += [args.loss, f"e{args.epochs}"]
    return "_".join(parts)


def setup_trainable(model, args):
    if args.init:              
        params = list(model.parameters())
        for p in params:
            p.requires_grad = True
        return [{"params": params, "lr": args.lr_pretrained}]

    out_params = list(model.head.parameters())          # classifier[4]
    out_ids = {id(p) for p in out_params}
    for p in model.parameters():
        p.requires_grad = args.train == "full"

    if args.train == "lastlayer":
        for p in out_params:
            p.requires_grad = True
        return [{"params": out_params, "lr": LR_NEW}]

    if args.train == "head":
        head_params = list(model.net.classifier.parameters())
        for p in head_params:
            p.requires_grad = True
        if args.head_init == "scratch":              
            return [{"params": head_params, "lr": LR_NEW}]
        aspp = [p for p in head_params if id(p) not in out_ids]
        return [{"params": aspp, "lr": args.lr_pretrained},
                {"params": out_params, "lr": LR_NEW}]

    rest = [p for p in model.parameters() if id(p) not in out_ids]     # full
    return [{"params": rest, "lr": args.lr_pretrained},
            {"params": out_params, "lr": LR_NEW}]


def set_train_mode(model, args):
    if args.train == "lastlayer":
        model.eval()
        return
    if args.train == "head":
        model.eval()
        model.net.classifier.train()
        return
    model.train()


def class_weights():
    pix = torch.zeros(NUM_CLASSES)
    img_pix = torch.zeros(NUM_CLASSES)
    for _, y in CamVid(split="train", transforms=get_transforms(train=False)):
        valid = y[y != IGNORE_INDEX]
        c = torch.bincount(valid, minlength=NUM_CLASSES).float()
        pix += c
        img_pix += (c > 0).float() * valid.numel()
    freq = pix / img_pix
    w = freq.median() / freq
    print("веса классов:", {n: round(v, 2) for n, v in zip(CLASSES, w.tolist())})
    return w


def make_criterion(args, device):
    if args.loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    if args.loss == "wce":
        return nn.CrossEntropyLoss(weight=class_weights().to(device), ignore_index=IGNORE_INDEX)
    import segmentation_models_pytorch as smp        
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    lovasz = smp.losses.LovaszLoss("multiclass", ignore_index=IGNORE_INDEX)
    return lambda logits, y: ce(logits, y) + lovasz(logits.float(), y)


def build(args, device):
    if args.init:
        model = load_model(args.init, device)
    else:
        model = build_model()
        if args.train == "head" and args.head_init == "scratch":
            model.net.classifier = DeepLabHead(2048, NUM_CLASSES)
    return model.to(device)


def train_epoch(model, train_dl, opt, sched, scaler, criterion, device, use_amp):
    loss_sum, n = 0.0, 0
    for x, y in train_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item()
        n += 1
    return loss_sum / n


def main():
    args = parse_args()
    name = args.name or default_name(args)
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    train_dl, val_dl = get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = build(args, device)
    groups = setup_trainable(model, args)
    total_p, trainable_p = count_params(model)
    print(f"[{name}] параметров: {total_p/1e6:.2f}M, обучаемых: {trainable_p/1e6:.4f}M")

    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / total_steps) ** 0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = make_criterion(args, device)

    history, best_miou, best_epoch = [], -1.0, -1
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_dl, opt, sched, scaler, criterion, device, use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0

        val, _ = evaluate(model, val_dl, device, amp=use_amp)
        row = {"epoch": epoch, "train_loss": train_loss, "val_miou": val["miou"],
               "val_pixel_acc": val["pixel_acc"], "epoch_time_s": epoch_time,
               "train_peak_mem_MB": peak_mem, "lr_new": opt.param_groups[-1]["lr"]}
        history.append(row)
        print(f"[{name}] ep {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"mIoU {val['miou']:.4f}  {epoch_time:.1f}s  {peak_mem:.0f}MB")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "miou": val["miou"],
                "args": vars(args)}
        torch.save(ckpt, out / "last.pt")
        if val["miou"] > best_miou:
            best_miou, best_epoch = val["miou"], epoch
            torch.save(ckpt, out / "best.pt")

    train_time = time.perf_counter() - t_start
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


    del opt, scaler, sched
    model.zero_grad(set_to_none=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    if args.init:                
        torch.save(model, out / "best_model.pt")
    val, cm = evaluate(model, val_dl, device, amp=use_amp)
    np.save(out / "confusion.npy", cm.numpy())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bench = benchmark(model, device)

    summary = {
        "name": name, "train": args.train, "init": args.init,
        "head_init": args.head_init if args.train == "head" else None,
        "loss": args.loss, "epochs": args.epochs, "batch_size": BATCH_SIZE,
        "lr_new": LR_NEW, "lr_pretrained": args.lr_pretrained,
        "weight_decay": WEIGHT_DECAY, "seed": SEED, "amp": USE_AMP,
        "trainable_params_M": trainable_p / 1e6,
        "best_epoch": best_epoch, "best_miou": val["miou"],
        "final_miou": history[-1]["val_miou"], "pixel_acc": val["pixel_acc"],
        "iou_per_class": dict(zip(CLASSES, val["iou"])),
        "train_time_min": train_time / 60,
        "mean_epoch_time_s": float(np.mean([r["epoch_time_s"] for r in history])),
        "train_peak_mem_MB": float(max(r["train_peak_mem_MB"] for r in history)),
        **bench,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{name}] лучший mIoU {val['miou']:.4f} (эпоха {best_epoch})")
    print(f"Сохранено в {out}/")


if __name__ == "__main__":
    main()

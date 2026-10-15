import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from camvid import CLASSES, IGNORE_INDEX, get_loaders
from seg_utils import build_model, evaluate, set_seed
from bench import benchmark, count_params

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "full"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--freeze-bn", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    return ap.parse_args()

def setup_trainable(model, args):
    head_params = list(model.head.parameters())
    if args.mode == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
        return [{"params": head_params, "lr": args.lr_head}]
    head_ids = {id(p) for p in head_params}
    rest = [p for p in model.parameters() if id(p) not in head_params]
    return [{"params": rest, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head}]

def set_train_mode(model, args):
    if args.mode == "head":
        model.eval()
        return
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def main():
    args = parse_args()
    name = args.name or args.mode 
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    train_dl, val_dl = get_loaders(batch_size=args.batch_size, num_workers=args.workers)
    model = build_model().to(device)
    groups = setup_trainable(model, args)
    total_p, trainable_p = count_params(model)

    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / total_steps) ** 0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    history, best_miou, best_epoch = [], -1.0, -1
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0

        val, _ = evaluate(model, val_dl, device, amp=use_amp)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_miou": val["miou"],
             "val_pixel_acc": val["pixel_acc"], "epoch_time_s": epoch_time,
             "train_peak_mem_MB": peak_mem, "lr_head": opt.param_groups[-1]["lr"]}
        history.append(row)
        print(f"[{name}] ep {epoch:3d}/{args.epochs} loss {row['train_loss']:.4f} "
             f"mIoU {val['miou']:.4f} {epoch_time:.1f}s {peak_mem:.0f}MB")

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

    del opt, scaler
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    val, cm = evaluate(model, val_dl, device, amp=use_amp)
    np.save(out / "confusion.npy", cm.numpy())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bench = benchmark(model, device)

    summary = {
        "name": name, "mode": args.mode, "epochs": args.epochs,
        "batch_size": args.batch_size, "seed": args.seed, "freeze_bn": args.freeze_bn,
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

if __name__ == "__main__":
    main()



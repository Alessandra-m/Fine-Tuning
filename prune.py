import argparse
import json
from pathlib import Path

import torch
import torch_pruning as tp

from bench import INPUT_SIZE, benchmark
from camvid import make_loader
from seg_utils import evaluate, get_device, load_model

DEFAULT_BASE = "runs/full_ce_lovasz_e50/best.pt"
ROUND_TO = 8           
OUT_DIR = Path("runs")


def prune_model(model, ratio, device):
    example_inputs = torch.randn(1, 3, *INPUT_SIZE, device=device)
    Pruner = getattr(tp.pruner, "BasePruner", None) or tp.pruner.MetaPruner  
    pruner = Pruner(
        model.eval(),
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[model.head],   
        round_to=ROUND_TO,
    )
    pruner.step()
    return model


def quality(model, loaders, device):
    res = {}
    for split, dl in loaders.items():
        m, _ = evaluate(model, dl, device)
        res[f"miou_{split}"] = m["miou"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="исходная модель")
    ap.add_argument("--ratio", type=float, required=True,
                    help="доля каналов, удаляемая в каждом слое")
    args = ap.parse_args()

    device = get_device()
    name = f"{Path(args.base).parent.name}_prune_l2_r{args.ratio}"
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    loaders = {"val": make_loader("val"), "test": make_loader("test")}

    model = load_model(args.base, device).eval()
    before = quality(model, loaders, device)
    print(f"[{name}] до прунинга:    {before}")

    model = prune_model(model, args.ratio, device)
    after = quality(model, loaders, device)
    print(f"[{name}] после прунинга: {after}")

    torch.save(model, out / "pruned.pt")

    bench = benchmark(model, device)
    print(json.dumps(bench, indent=2, ensure_ascii=False))

    summary = {"name": name, "base": args.base, "ratio": args.ratio,
               "before": before, "after_prune": after, **bench}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Сохранено в {out}/")


if __name__ == "__main__":
    main()

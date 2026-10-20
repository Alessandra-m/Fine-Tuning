import torch
import torch.nn as nn
import torch_pruning as tp

from bench import INPUT_SIZE
from seg_utils import get_device, load_model

CKPT = "runs/full_ce_lovasz_e50/best.pt"


def main():
    device = get_device()
    model = load_model(CKPT, device).eval()
    example_inputs = torch.randn(1, 3, *INPUT_SIZE, device=device)
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    names = {m: n for n, m in model.named_modules()}
    sizes = []
    for group in dg.get_all_groups(ignored_layers=[model.head], root_module_types=[nn.Conv2d]):
        root = group[0].dep.target.module                      
        layers = {names.get(item.dep.target.module) for item in group} - {None}
        sizes.append(len(layers))
        print(f"{len(layers):3d} слоёв | {names[root]}")

    print(f"\nВсего групп: {len(sizes)}")
    for s in sorted(set(sizes)):
        print(f"  по {s:2d} слоёв: {sizes.count(s)} групп")


if __name__ == "__main__":
    main()
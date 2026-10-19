import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp

from bench import INPUT_SIZE, count_gmacs, count_params, measure_latency
from prune import prune_model
from seg_utils import get_device, load_model

CKPT = "runs/full_ce_lovasz_e50/best.pt"
RATIO = 0.5


def show_groups(model, device):
    example_inputs = torch.randn(1, 3, *INPUT_SIZE, device=device)
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    backbone = model.net.backbone
    aspp = model.net.classifier[0]
    cases = {
        "layer1[0].conv1": backbone.layer1[0].conv1,
        "layer1[0].conv3": backbone.layer1[0].conv3,
        "ASPP, ветвь 3x3 r=12": aspp.convs[1][0],
    }
    for title, layer in cases.items():
        group = dg.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[0])
        print(f"{title}: {len(group)} зависимостей")
        print(group)


def make_zeroed(model, ratio):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m is not model.head:
            prune.ln_structured(m, name="weight", amount=ratio, n=2, dim=0)
            prune.remove(m, "weight")   
    return model


def count_nonzero(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def report(name, model, device):
    total, _ = count_params(model)
    lat, _ = measure_latency(model, device, amp=False)
    w = model.net.backbone.layer1[0].conv1.weight
    print(f"{name:8s} | conv1.weight {str(tuple(w.shape)):16s} | "
          f"params {total/1e6:6.2f}M | ненулевых {count_nonzero(model)/1e6:6.2f}M | "
          f"{count_gmacs(model, device):6.1f} GMACs | {lat:5.1f} мс")


def main():
    device = get_device()
    torch.backends.cudnn.benchmark = True

    show_groups(load_model(CKPT, device).eval(), device)

    print(f"Обнуление и удаление {RATIO:.0%} каналов в каждой свёртке\n")
    models = {
        "orig": load_model(CKPT, device).eval(),
        "zeroed": make_zeroed(load_model(CKPT, device).eval(), RATIO),
        "pruned": prune_model(load_model(CKPT, device).eval(), RATIO, device),
    }

    measure_latency(models["orig"], device, iters=50)
    for name, m in models.items():
        report(name, m, device)


if __name__ == "__main__":
    main()

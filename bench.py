import argparse
import json
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

INPUT_SIZE = (360, 480)  


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def count_gmacs(model, device, input_size=INPUT_SIZE):
    model.eval()
    x = torch.randn(1, 3, *input_size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9  # FLOPs = 2 * MACs


@torch.no_grad()
def measure_latency(model, device, input_size=INPUT_SIZE, amp=False,
                    warmup=20, iters=100):
    model.eval()
    x = torch.randn(1, 3, *input_size, device=device)
    use_amp = amp and device.type == "cuda"

    def run():
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            model(x)

    for _ in range(warmup):
        run()
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


@torch.no_grad()
def measure_infer_memory(model, device, input_size=INPUT_SIZE, amp=False):
    """Пиковая память инференса (веса + активации), МБ."""
    if device.type != "cuda":
        return None
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1, 3, *input_size, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def benchmark(model, device, input_size=INPUT_SIZE, iters=100):
    torch.backends.cudnn.benchmark = True
    model = model.to(device).eval()
    total, _ = count_params(model)
    lat32, lat32_p90 = measure_latency(model, device, input_size, amp=False, iters=iters)
    lat16, lat16_p90 = measure_latency(model, device, input_size, amp=True, iters=iters)
    return {
        "params_M": total / 1e6,
        "gmacs": count_gmacs(model, device, input_size),
        "latency_fp32_ms": lat32,
        "latency_fp32_p90_ms": lat32_p90,
        "latency_fp16_ms": lat16,
        "latency_fp16_p90_ms": lat16_p90,
        "infer_mem_fp32_MB": measure_infer_memory(model, device, input_size, amp=False),
        "input_size": list(input_size),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


if __name__ == "__main__":
    from seg_utils import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = benchmark(load_model(args.ckpt, dev), dev, iters=args.iters)
    print(json.dumps(res, indent=2, ensure_ascii=False))

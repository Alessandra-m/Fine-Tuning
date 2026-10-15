import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights)

from camvid import NUM_CLASSES, IGNORE_INDEX

class SegModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]

    @property
    def head(self):
        return self.net.classifier[4]


def build_model(num_classes = NUM_CLASSES, pretrained = True):
    if pretrained:
        net = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    else:
        net = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)

    net.aux_classifier = None
    net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, num_classes, kernel_size=1)
    return SegModel(net)

def load_model(path, device="cpu"):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = build_model(pretrained=False)
        model.load_state_dict(obj["model"])
    return model.to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def update_cm(cm, pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    valid = target != ignore_index
    idx = target[valid] * num_classes + pred[valid]
    cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm

def iou_from_cm(cm):
    cm = cm.double()
    tp = cm.diag()
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = tp / denom
    iou[denom == 0] = float("nan")
    return iou

def metrics_from_cm(cm):
    iou = iou_from_cm(cm)
    return {"miou": float(np.nanmean(iou.numpy())), 
            "pixel_acc": float(cm.diag().sum() / cm.sum().clamp(min=1)),
            "iou": [float(v) for v in iou],
            }

@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        update_cm(cm, logits.argmax(1), y)
    cm = cm.cpu()
    return metrics_from_cm(cm), cm






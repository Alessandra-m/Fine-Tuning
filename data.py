import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

ROOT = "data"
IMAGE_SIZE = 256
CLASSES = ("pet", "background", "border")
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255
TRIMAP_OFFSET = 1  # trimap values in the dataset: 1 = pet, 2 = background, 3 = border
MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics expected by the pretrained weights
STD = (0.229, 0.224, 0.225)


# transforms

def train_transform(size=IMAGE_SIZE):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX}),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        T.Normalize(MEAN, STD),
    ])


def val_transform(size=IMAGE_SIZE):
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        T.Normalize(MEAN, STD),
    ])


class PetSegmentation(Dataset):
    """Oxford-IIIT Pet with masks remapped to 0 = pet, 1 = background, 2 = border."""

    def __init__(self, split, transform, root=ROOT, download=True):
        self.base = OxfordIIITPet(root, split=split, target_types="segmentation", download=download)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, trimap = self.base[idx]
        image = tv_tensors.Image(F.pil_to_tensor(image))
        mask = tv_tensors.Mask(F.pil_to_tensor(trimap) - TRIMAP_OFFSET)
        image, mask = self.transform(image, mask)
        return image, mask[0]


def make_loaders(batch_size=8, num_workers=4, size=IMAGE_SIZE):
    train_set = PetSegmentation("trainval", train_transform(size))
    val_set = PetSegmentation("test", val_transform(size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def check_classes(dataset, n=200):
    """Verify the raw trimap values and the remapped masks on the first n samples."""
    raw_values = set()
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    ignored = 0
    for i in range(min(n, len(dataset))):
        _, trimap = dataset.base[i]
        raw_values.update(np.unique(np.array(trimap)).tolist())

        _, mask = dataset[i]
        values = set(mask.unique().tolist())
        assert values <= set(range(NUM_CLASSES)) | {IGNORE_INDEX}, f"sample {i}: mask values {values}"
        valid = mask[mask != IGNORE_INDEX]
        counts += torch.bincount(valid.flatten(), minlength=NUM_CLASSES)
        ignored += (mask == IGNORE_INDEX).sum().item()

    assert raw_values <= {1, 2, 3}, f"unexpected trimap values {sorted(raw_values)}"
    total = counts.sum().item() + ignored
    print(f"raw trimap values {sorted(raw_values)}  ->  classes {list(range(NUM_CLASSES))}")
    for name, count in zip(CLASSES, counts.tolist()):
        print(f"  {name:<11} {count / total:6.1%}")
    print(f"  {'ignore':<11} {ignored / total:6.1%}")


# preview

PALETTE = np.array([[1.0, 0.55, 0.0], [0.0, 0.0, 0.0], [0.0, 0.8, 1.0]])  # pet, background, border


def denormalize(image):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image, mask, alpha=0.5):
    rgb = denormalize(image)
    out = rgb.copy()
    m = mask.numpy()
    for cls in (0, 2):  # color pet and border, leave background as is
        sel = m == cls
        out[sel] = (1 - alpha) * rgb[sel] + alpha * PALETTE[cls]
    out[m == IGNORE_INDEX] = 0.5
    return out


def save_augmentation_preview(dataset, idx, path, n=4):
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for k in range(n):
        image, mask = dataset[idx]  # a new random augmentation on every call
        axes[0, k].imshow(denormalize(image))
        axes[0, k].set_title(f"augmentation {k + 1}", fontsize=9)
        axes[1, k].imshow(overlay(image, mask))
        axes[1, k].set_title("mask overlay", fontsize=9)
        for ax in axes[:, k]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    train_set = PetSegmentation("trainval", train_transform())
    val_set = PetSegmentation("test", val_transform())
    print(f"train {len(train_set)} images, val {len(val_set)} images")

    print("\n=== class check: train (with augmentations) ===")
    check_classes(train_set)
    print("\n=== class check: val ===")
    check_classes(val_set)

    save_augmentation_preview(train_set, idx=0, path="augmentations.png")

    train_loader, _ = make_loaders()
    images, masks = next(iter(train_loader))
    print(f"\nbatch: images {list(images.shape)} {images.dtype}, masks {list(masks.shape)} {masks.dtype}")
    print("saved augmentations.png")


if __name__ == "__main__":
    main()
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
NUM_CLASSES = len(CLASSES)        
IGNORE_INDEX = 11                

PALETTE = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128),
           (60, 40, 222), (128, 128, 0), (192, 128, 128), (64, 64, 128),
           (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)]

MEAN = [0.485, 0.456, 0.406]     
STD = [0.229, 0.224, 0.225]

TRAIN_CROP = (352, 480)  


def get_transforms(train: bool):
    to_float = v2.ToDtype({tv_tensors.Image: torch.float32,
                           tv_tensors.Mask: torch.int64}, scale=True)
    if not train:
        return v2.Compose([to_float, v2.Normalize(MEAN, STD)])

    return v2.Compose([
        v2.RandomResizedCrop(TRAIN_CROP, scale=(0.5, 1.0), ratio=(1.2, 1.5), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomAffine(degrees=10, fill={tv_tensors.Image: 0, 
                                                          tv_tensors.Mask: IGNORE_INDEX},)], p=0.3),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        to_float,
        v2.Normalize(MEAN, STD),
    ])


class CamVid(Dataset):
    def __init__(self, root="data/CamVid", split="train", transforms=None):
        root = Path(root)
        self.images = sorted((root / split).glob("*.png"))
        self.masks = sorted((root / f"{split}annot").glob("*.png"))
        assert len(self.images) > 0, f"нет изображений в {root / split}"
        assert [p.name for p in self.images] == [p.name for p in self.masks], \
            f"{split}: имена картинок и масок не совпадают"
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = tv_tensors.Image(read_image(str(self.images[i]), ImageReadMode.RGB))
        mask = tv_tensors.Mask(read_image(str(self.masks[i]), ImageReadMode.GRAY)[0])
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask


def get_loaders(root="data/CamVid", batch_size=4, num_workers=4):
    train_ds = CamVid(root, "train", get_transforms(train=True))
    val_ds = CamVid(root, "val", get_transforms(train=False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return train_dl, val_dl









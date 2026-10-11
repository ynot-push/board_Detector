import pathlib
import random
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

IN_H, IN_W = 256, 448
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

_base = transforms.Compose([
    transforms.Resize((IN_H, IN_W)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def preprocess(pil_img):
    return _base(pil_img.convert("RGB")).clone()

class BoardDataset(Dataset):
    def __init__(self, csv_path, image_dir, train=False):
        self.df = pd.read_csv(csv_path)
        self.image_dir = pathlib.Path(image_dir)
        self.train = train
        self.jitter = transforms.ColorJitter(0.35, 0.35, 0.3, 0.03)

    def __len__(self):
        return len(self.df)

    def _augment(self, img, has, box):
        w, h = img.size

        if random.random() < 0.6:
            if has:
                x1, y1, x2, y2 = box

                x1 = max(0, min(float(x1), w - 1))
                y1 = max(0, min(float(y1), h - 1))
                x2 = max(x1 + 1, min(float(x2), w))
                y2 = max(y1 + 1, min(float(y2), h))

                left = int(random.uniform(0, x1))
                top = int(random.uniform(0, y1))
                right = int(random.uniform(x2, w))
                bottom = int(random.uniform(y2, h))

                right = max(right, left + 1)
                bottom = max(bottom, top + 1)

                img = img.crop((left, top, right, bottom))
                box = (x1 - left, y1 - top, x2 - left, y2 - top)
            else:
                cw = random.uniform(0.4, 1.0) * w
                ch = random.uniform(0.4, 1.0) * h

                cw = max(1, min(cw, w))
                ch = max(1, min(ch, h))

                left = int(random.uniform(0, max(0, w - cw)))
                top = int(random.uniform(0, max(0, h - ch)))

                right = max(left + 1, int(left + cw))
                bottom = max(top + 1, int(top + ch))

                right = min(right, w)
                bottom = min(bottom, h)

                img = img.crop((left, top, right, bottom))

            w, h = img.size

        if random.random() < 0.7:
            img = self.jitter(img)

        if random.random() < 0.10:
            img = img.convert("L").convert("RGB")

        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.2)))

        return img, box, w, h

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(
            self.image_dir / row["filename"]
        ).convert("RGB")

        cached_w, cached_h = img.size

        has = bool(row["has_board"])

        if has:
            sx = cached_w / float(row["screen_width"])
            sy = cached_h / float(row["screen_height"])

            box = (
                float(row["x1"]) * sx,
                float(row["y1"]) * sy,
                float(row["x2"]) * sx,
                float(row["y2"]) * sy,
            )
        else:
            box = (0.0, 0.0, 0.0, 0.0)

        w, h = cached_w, cached_h

        if self.train:
            img, box, w, h = self._augment(img, has, box)

        if has:
            target_box = torch.tensor(
                [
                    box[0] / w,
                    box[1] / h,
                    box[2] / w,
                    box[3] / h,
                ],
                dtype=torch.float32,
            ).clamp(0, 1)
        else:
            target_box = torch.zeros(4, dtype=torch.float32)

        flag = torch.tensor(
            [float(has)],
            dtype=torch.float32
        )

        return preprocess(img), target_box, flag

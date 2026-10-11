import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import BoardDataset
from model import BoardDetectorCNN, detector_loss


BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
SEED = 42

NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL = 8

IMAGE_DIR = "dataset/images_448"
CSV_PATH = "dataset/labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    "mobilenet_v3_large",
    "efficientnet_b0",
    "mobilenet_v2",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    imgs, boxes, flags = zip(*batch)

    return (
        torch.stack(imgs, dim=0),
        torch.stack(boxes, dim=0),
        torch.stack(flags, dim=0),
    )


def box_iou(pred, target):
    px1, py1, px2, py2 = pred.unbind(1)
    tx1, ty1, tx2, ty2 = target.unbind(1)

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)

    inter = iw * ih

    p_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    t_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    union = p_area + t_area - inter

    return inter / union.clamp(min=1e-7)


def model_size_mb(model):
    params = sum(p.numel() for p in model.parameters())

    buffers = sum(b.numel() for b in model.buffers())

    size_bytes = (params + buffers) * 4

    return params, size_bytes / (1024 ** 2)


def make_datasets():
    full_ds = BoardDataset(
        CSV_PATH,
        IMAGE_DIR,
        train=True,
    )

    n = len(full_ds)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    indices = torch.randperm(
        n,
        generator=generator,
    ).tolist()

    split = int(n * 0.9)

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = Subset(
        full_ds,
        train_indices,
    )

    val_base = BoardDataset(
        CSV_PATH,
        IMAGE_DIR,
        train=False,
    )

    val_ds = Subset(
        val_base,
        val_indices,
    )

    return train_ds, val_ds


def make_loaders(train_ds, val_ds):
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=NUM_WORKERS_TRAIN > 0,
        prefetch_factor=4 if NUM_WORKERS_TRAIN > 0 else None,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
        persistent_workers=NUM_WORKERS_VAL > 0,
        prefetch_factor=4 if NUM_WORKERS_VAL > 0 else None,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_correct = 0
    total_iou = 0.0
    total_board_samples = 0

    with torch.no_grad():
        for img, box, flag in loader:
            img = img.to(
                DEVICE,
                non_blocking=True,
            )

            box = box.to(
                DEVICE,
                non_blocking=True,
            )

            flag = flag.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                flag_pred, box_pred = model(img)

                loss = detector_loss(
                    flag_pred,
                    box_pred,
                    flag,
                    box,
                )

            batch_size = img.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred_flag = (
                torch.sigmoid(flag_pred) >= 0.5
            ).float()

            total_correct += (
                pred_flag == flag
            ).sum().item()

            mask = flag.squeeze(1) > 0.5

            if mask.any():
                ious = box_iou(
                    box_pred[mask],
                    box[mask],
                )

                total_iou += ious.sum().item()
                total_board_samples += mask.sum().item()

    avg_loss = total_loss / total_samples

    flag_acc = total_correct / total_samples

    mean_iou = (
        total_iou / total_board_samples
        if total_board_samples > 0
        else 0.0
    )

    return avg_loss, flag_acc, mean_iou


def run_experiment(
    model_name,
    train_loader,
    val_loader,
):
    print()
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    model = BoardDetectorCNN(
        pretrained=True,
        backbone=model_name,
        reduce_ch=32,
        hidden=128,
    ).to(DEVICE)

    params, size_mb = model_size_mb(model)

    print(f"Parameters: {params:,}")
    print(f"FP32 size: {size_mb:.2f} MB")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda",
    )

    os.makedirs(
        "experiments",
        exist_ok=True,
    )

    best_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        samples = 0

        start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False,
        )

        for img, box, flag in pbar:
            img = img.to(
                DEVICE,
                non_blocking=True,
            )

            box = box.to(
                DEVICE,
                non_blocking=True,
            )

            flag = flag.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                flag_pred, box_pred = model(img)

                loss = detector_loss(
                    flag_pred,
                    box_pred,
                    flag,
                    box,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            batch_size = img.size(0)

            epoch_loss += loss.item() * batch_size
            samples += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        scheduler.step()

        train_loss = epoch_loss / samples

        val_loss, flag_acc, mean_iou = evaluate(
            model,
            val_loader,
        )

        epoch_time = time.time() - start_time

        if mean_iou > best_iou:
            best_iou = mean_iou

            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "params": params,
                },
                f"experiments/{model_name}_best.pt",
            )

        torch.save(
            {
                "model": model.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "best_iou": best_iou,
                "params": params,
            },
            f"experiments/{model_name}_last.pt",
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Flag Acc: {flag_acc:.4f} | "
            f"Mean IoU: {mean_iou:.4f} | "
            f"Best IoU: {best_iou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

    del model
    torch.cuda.empty_cache()

    return best_iou


def main():
    seed_everything(SEED)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds = make_datasets()

    print(
        f"train: {len(train_ds)}, "
        f"val: {len(val_ds)}, "
        f"device: {DEVICE}"
    )

    print(f"images: {IMAGE_DIR}")

    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
    )

    results = {}

    for model_name in MODELS:
        best_iou = run_experiment(
            model_name,
            train_loader,
            val_loader,
        )

        results[model_name] = best_iou

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for model_name, best_iou in results.items():
        print(
            f"{model_name:25s} "
            f"Best IoU: {best_iou:.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

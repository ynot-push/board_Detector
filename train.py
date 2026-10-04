import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from dataset import BoardDataset
from model import BoardDetectorCNN
from tqdm import tqdm
import time

# ---- config ----
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
SEED = 42
LAMBDA = 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(board_logit, box_pred, box_true, flag):
    flag_loss = nn.BCEWithLogitsLoss()(board_logit, flag)

    box_error = F.l1_loss(box_pred, box_true, reduction="none")
    masked_box_error = box_error * flag
    box_loss = masked_box_error.sum() / (flag.sum() * 4 + 1e-6)

    return flag_loss + LAMBDA * box_loss


def compute_iou(box_pred, box_true):
    pred_x1, pred_y1, pred_x2, pred_y2 = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
    true_x1, true_y1, true_x2, true_y2 = box_true[:, 0], box_true[:, 1], box_true[:, 2], box_true[:, 3]
    inter_w = (torch.minimum(pred_x2, true_x2) - torch.maximum(pred_x1, true_x1)).clamp(min=0)
    inter_h = (torch.minimum(pred_y2, true_y2) - torch.maximum(pred_y1, true_y1)).clamp(min=0)
    inter_area = inter_w * inter_h
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)

    union = pred_area + true_area - inter_area
    return inter_area / (union + 1e-6)


if __name__ == "__main__":
    ds = BoardDataset("dataset/labels.csv", "dataset/images")

    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, device: {DEVICE}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BoardDetectorCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_iou = 0.0


    for epoch in range(EPOCHS):

        # Added: measure how long this epoch takes
        t0 = time.time()

        model.train()

        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for img, box, flag in pbar:

            img = img.to(DEVICE)
            flag = flag.to(DEVICE)
            box = box.to(DEVICE)

            logit, box_pred = model(img)

            loss = compute_loss(
                logit,
                box_pred,
                box,
                flag
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            # Added: show current loss on the tqdm bar
            # Track loss
            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches


        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_batches = 0

        total_correct = 0
        total_samples = 0

        iou_sum = 0.0
        iou_count = 0

        with torch.no_grad():

            for img, box, flag in tqdm(val_loader,desc=f"Epoch {epoch+1}/{EPOCHS}",leave=False):

                # Move data to device
                img = img.to(DEVICE)
                flag = flag.to(DEVICE)
                box = box.to(DEVICE)

                # Forward pass
                logit, box_pred = model(img)

                # Calculate loss
                loss = compute_loss(
                    logit,
                    box_pred,
                    box,
                    flag
                )

                val_loss += loss.item()
                val_batches += 1

                # ------------------------------------------------
                # Flag accuracy
                # ------------------------------------------------

                predicted_flag = (logit > 0)

                correct = (predicted_flag == flag.bool()).sum().item()

                total_correct += correct
                total_samples += flag.numel()

                # ------------------------------------------------
                # IoU only for images containing a board
                # ------------------------------------------------

                board_mask = flag.squeeze(1).bool()

                if board_mask.any():

                    board_iou = compute_iou(
                        box_pred[board_mask],
                        box[board_mask]
                    )

                    iou_sum += board_iou.sum().item()
                    iou_count += board_iou.numel()


        # Average validation metrics

        val_loss /= val_batches

        flag_accuracy = total_correct / total_samples

        mean_iou = (
            iou_sum / iou_count
            if iou_count > 0
            else 0.0
        )


        scheduler.step()


        # Added: update best IoU and save best checkpoint
        if mean_iou > best_iou:

            best_iou = mean_iou

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_iou": best_iou,
                },
                "best_detector.pt"
            )


        # Added: save checkpoint after every epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou,
            },
            "last.pt"
        )


        # Added: calculate epoch duration
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Flag Acc: {flag_accuracy:.4f} | "
            f"Mean IoU: {mean_iou:.4f} | "
            f"Best IoU: {best_iou:.4f} | "
            f"scheduler: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
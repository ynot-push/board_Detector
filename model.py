import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    mobilenet_v3_large,
    efficientnet_b0,
    mobilenet_v2,
)

IN_H, IN_W = 256, 448


def _make_backbone(name, pretrained):
    if name == "mobilenet_v3_large":
        m = mobilenet_v3_large(
            weights=(
                MobileNet_V3_Large_Weights.IMAGENET1K_V1
                if pretrained else None
            )
        )
        return m.features

    if name == "efficientnet_b0":
        m = efficientnet_b0(
            weights=(
                EfficientNet_B0_Weights.IMAGENET1K_V1
                if pretrained else None
            )
        )
        return m.features

    if name == "mobilenet_v2":
        m = mobilenet_v2(
            weights=(
                MobileNet_V2_Weights.IMAGENET1K_V1
                if pretrained else None
            )
        )
        return m.features

    raise ValueError(f"unknown backbone {name!r}")


class BoardDetectorCNN(nn.Module):
    def __init__(
        self,
        pretrained=True,
        backbone="mobilenet_v3_large",
        reduce_ch=32,
        hidden=128,
    ):
        super().__init__()

        self.backbone = _make_backbone(
            backbone,
            pretrained,
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                IN_H,
                IN_W,
            )

            feature = self.backbone(dummy)

        _, ch, fh, fw = feature.shape

        self.reduce = nn.Sequential(
            nn.Conv2d(
                ch,
                reduce_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(reduce_ch),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(
                reduce_ch * fh * fw,
                hidden,
            ),
            nn.ReLU(inplace=True),
        )

        self.fc_flag = nn.Linear(
            hidden,
            1,
        )

        self.fc_box = nn.Linear(
            hidden,
            4,
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.reduce(f)
        f = self.fc(f)

        flag_logits = self.fc_flag(f)

        cx, cy, w, h = torch.sigmoid(
            self.fc_box(f)
        ).unbind(1)

        box = torch.stack(
            [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
            ],
            dim=1,
        ).clamp(0, 1)

        return flag_logits, box


def giou_loss(p, t, eps=1e-7):
    px1, py1, px2, py2 = p.unbind(1)
    tx1, ty1, tx2, ty2 = t.unbind(1)

    pa = (
        (px2 - px1).clamp(min=0)
        * (py2 - py1).clamp(min=0)
    )

    ta = (
        (tx2 - tx1).clamp(min=0)
        * (ty2 - ty1).clamp(min=0)
    )

    iw = (
        torch.min(px2, tx2)
        - torch.max(px1, tx1)
    ).clamp(min=0)

    ih = (
        torch.min(py2, ty2)
        - torch.max(py1, ty1)
    ).clamp(min=0)

    inter = iw * ih

    union = pa + ta - inter + eps

    cw = (
        torch.max(px2, tx2)
        - torch.min(px1, tx1)
    )

    ch = (
        torch.max(py2, ty2)
        - torch.min(py1, ty1)
    )

    carea = cw * ch + eps

    giou = (
        inter / union
        - (carea - union) / carea
    )

    return 1 - giou


def detector_loss(
    flag_pred,
    box_pred,
    flag_true,
    box_true,
    w_l1=5.0,
    w_giou=2.0,
):
    bce = F.binary_cross_entropy_with_logits(
        flag_pred,
        flag_true,
    )

    mask = flag_true.squeeze(1) > 0.5

    if mask.any():
        l1 = F.l1_loss(
            box_pred[mask],
            box_true[mask],
        )

        giou = giou_loss(
            box_pred[mask],
            box_true[mask],
        ).mean()

    else:
        l1 = box_pred.sum() * 0.0
        giou = box_pred.sum() * 0.0

    return (
        bce
        + w_l1 * l1
        + w_giou * giou
    )

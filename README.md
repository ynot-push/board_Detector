# Chessboard Detector

A CNN-based detector for identifying chessboards in screenshots from online chess websites and predicting the board's bounding box.

The project combines:

- binary classification: whether a chessboard is present
- bounding box regression: where the chessboard is located in the image

This is a lightweight object detection pipeline tailored to board localization in browser screenshots.

## Project Goal

The model is designed to detect a square chessboard within screenshots taken from different online chess platforms. In practice, the board may appear in different positions, scales, and visual styles, so the detector needs to be robust to background variation, UI clutter, and synthetic image augmentation.

## Model Architecture

The model is a custom CNN with a pretrained backbone and two output heads:

```text
Input Image
     │
     ▼
 CNN Backbone (MobileNet / EfficientNet)
     │
     ├──► Board Presence Classification
     │
     └──► Bounding Box Regression
```

The backbone options in the repository are:

- MobileNetV3 Large
- EfficientNet-B0
- MobileNetV2

The model outputs:

- a single logit for board/no-board classification
- a 4-value normalized bounding box: [x1, y1, x2, y2]

Each box is normalized to the image size and represented as:

- x1 = left edge
- y1 = top edge
- x2 = right edge
- y2 = bottom edge

Because the chessboard is square, the model predicts the board's full enclosing box.

## Repository Structure

```text
.
├── README.md
├── model.py
├── dataset.py
├── train.py
├── generate_board_Screenshots.py
├── mobilenet_v3_large_best.pt
└── dataset/
    ├── images_448/
    └── labels.csv
```

### Key files

- `model.py`: defines the detector model and loss function
- `dataset.py`: dataset loader and augmentation pipeline
- `train.py`: training, validation, model selection, and experiment loop
- `generate_board_Screenshots.py`: produces synthetic dataset screenshots and labels

## Dataset

The dataset is generated synthetically using `generate_board_Screenshots.py` and stored as image files plus a CSV labels file.

The dataset includes:

- board/no-board labels
- board bounding box coordinates for present boards
- image variations to simulate real website screenshots

The data pipeline applies several augmentations in `dataset.py`, including:

- random crop and resize
- color jitter
- grayscale conversion
- Gaussian blur
- normalization to ImageNet statistics

## Training

The training script (`train.py`) performs supervised learning using a combined loss:

- binary cross-entropy for board presence
- L1 loss for bounding box regression
- GIoU loss for geometric alignment

### Training configuration

```python
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
SEED = 42
```

The script evaluates:

- classification accuracy
- mean IoU over board-positive samples

It also compares multiple backbones:

- `mobilenet_v3_large`
- `efficientnet_b0`
- `mobilenet_v2`

## Usage

### Install dependencies

```bash
pip install torch torchvision pandas pillow tqdm
```

### Train the model

```bash
python train.py
```

This will create experiment checkpoints in the `experiments/` directory.

### Download or prepare the dataset

The repository expects a dataset folder like:

```text
dataset/
├── images_448/
└── labels.csv
```

If you are generating the synthetic data, run:

```bash
python generate_board_Screenshots.py
```

## Results

The project reports strong detection performance during training:
- MobileNetV3 Large:
  
     - Board detection accuracy: 100%
     - Mean IoU: ~0.991

 - EfficientNet-B0:
        - Board detection accuracy: 100%
        - Mean IoU: ~0.991
   
 - MobileNetV2:
      - Board detection accuracy: 100%
        - Mean IoU: ~0.993


These numbers indicate that the model is highly effective at detecting chessboards and localizing them in screenshot-based inputs.

## Notes

This project is useful as a focused computer vision task for web screenshot board detection. It demonstrates a practical implementation of:

- binary classification + localization
- pretrained backbones for efficient training
- synthetic dataset generation
- custom multi-task loss design

## License

This project does not currently specify a license file. If you plan to publish or reuse it publicly, consider adding an explicit open-source license.


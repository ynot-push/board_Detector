# Chessboard Detector

A CNN model for detecting chessboards in screenshots from different online chess websites.

The model performs two tasks:

- Board detection: determines whether a chessboard is present in the image.
- Bounding box regression: predicts the location of the chessboard.

## Model

The model is a custom CNN with two heads:

```text
Input Image
     │
     ▼
 CNN Feature Extractor
     │
     ├──► Board / No Board
     │
     └──► Bounding Box
```

The bounding box is represented as:

- x1 = left edge
- y1 = top edge
- x2 = right edge
- y2 = bottom edge

Since the chessboard is square, the box corresponds to the board's full square outline.

## Dataset

The screenshots in the dataset were synthetically generated using [generate_board_Screenshots.py](generate_board_Screenshots.py), which was created by Claude 😒.

## Results

On my side, training achieved:

- Board detection accuracy: 100%
- Mean IoU: ~0.97

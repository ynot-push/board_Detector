# Chessboard Detector

A CNN-based model for detecting chessboards in screenshots from various online chess websites.

The system performs two tasks:

- Board detection: determines whether a chessboard is present in the image.
- Bounding box regression: predicts the location of the chessboard within the screenshot.

## Overview

This project is designed for board localization in web-based chess screenshots, where the board may appear at different positions and scales. The model outputs a binary classification for board presence and a bounding box for the board region.

## Model Architecture

The model is a custom CNN with two output heads:

```text
Input Image
     │
     ▼
 CNN Feature Extractor
     │
     ├──► Board / No Board
     │
     └──► Bounding Box Regression
```

The bounding box is represented as:

- x1 = left edge
- y1 = top edge
- x2 = right edge
- y2 = bottom edge

Because the chessboard is square, the box corresponds to the full board region.

## Dataset

The training data consists of synthetic screenshots generated with [generate_board_Screenshots.py](generate_board_Screenshots.py).

This dataset was created to simulate chessboard appearances across different online chess platforms and visual conditions.

## Results

The model achieved strong performance during training:

- Board detection accuracy: 100%
- Mean IoU: ~0.97

## Notes

This project was built as a lightweight chessboard detection pipeline and demonstrates that a CNN can reliably identify board regions in screenshots.


# Chessboard Detector

a CNN model for detecting a chessboard inside screenshots of the different online chess websites

The model performs two tasks:

* **Board detection:** to determines whether a chessboard is present.
* **Bounding box regression:** predicts the location of the chessboard.

## Model

The model is a custom CNN with two  heads:

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

the bounding box is represented by : x1 = left edge , y1 = top edge , x2 = right edge , y2 = bottom edge (since the the chess board is a square)



## Dataset

the screenshots in the dataset used  was synthetically generated [](generate_board_Screenshots.py) which was created by claude 😒.
## Results

On my side training i got :

* **Board detection:** 100% accuracy
* **Mean IoU:** ~0.97



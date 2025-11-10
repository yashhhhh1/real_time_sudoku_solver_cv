# Real-Time Sudoku Solver (Computer Vision & CNN)

A real-time Sudoku board detector, digit recognizer and solver that reads a Sudoku puzzle from your webcam, solves it and overlays the solution back onto the video stream.

This project detects a Sudoku grid in a video frame, extracts each cell, recognizes digits using a small CNN, solves the puzzle using a Best-First search solver, and writes the solution back onto the frame in real time.

## Features
- Detects the largest Sudoku-like contour from a webcam frame and warps it to a bird's-eye view.
- Cleans and segments the board into 9x9 cells and recognizes digits using a pretrained CNN.
- Solves the Sudoku using an efficient Best-First search algorithm.
- Overlays the solved digits onto the original frame and displays the video in real time.

## Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the solver (will use your default webcam):

```bash
python3 main.py
```

Press `q` in the display window to quit.

Notes:
- The code expects a webcam accessible at index 0. If your camera is on another index, edit `main.py` (cv2.VideoCapture(0)).
- `main.py` is the primary entry point and already loads a pretrained model (`digitRecognition.h5`) for digit recognition.

## Files in this repository
- `main.py` — Entry point. Opens the webcam, loads the CNN model weights and repeatedly calls the real-time recognition routine.
- `RealTimeSudokuSolver.py` — Core image-processing and orchestration code. Detects the board, extracts cells, prepares images for the CNN, calls the solver, and overlays the result.
- `digitRecognition.py` — Keras code used to train the CNN that recognizes digits 1–9. It expects training data in a `DigitImages/` directory with subfolders `1`, `2`, ..., `9`. Running this script will train and write `digitRecognition.h5`.
- `digitRecognition.h5` — Pretrained CNN weights (used by `main.py`). The repository contains the saved weights so you don't need to train from scratch.
- `sudokuSolver.py` — Sudoku solver that implements a Best-First search heuristic and the helper functions used by `RealTimeSudokuSolver.py`.
- `requirements.txt` — Minimal Python dependencies inferred from the code (numpy, opencv-python, tensorflow, keras, scipy, h5py).

## Training your own digit recognizer

If you want to re-train the digit recognizer or adapt it to your own digit data:

1. Prepare the dataset structure:

```
DigitImages/
	1/
	2/
	...
	9/
```

Each folder should contain grayscale images of the digit centered in an image. The `digitRecognition.py` script performs centralization and normalization before training.

2. Run training (this can take time and requires enough RAM/CPU/GPU):

```bash
python3 digitRecognition.py
```

This saves weights to `digitRecognition.h5` which `main.py` expects.

## Troubleshooting and tips
- If the solver never finds a board or the overlay appears incorrectly:
	- Ensure the Sudoku board is the largest prominent contour in the camera frame (place it on a contrasting background).
	- Increase camera resolution or move closer so digits are clearly visible.
	- Try different lighting; shadows impact thresholding.

- If digit recognition is poor:
	- Re-train `digitRecognition.py` with more representative digit images (same resolution and similar pre-processing).
	- Verify that your training images are centered and have similar contrast to the webcam crops.

- TensorFlow/Keras errors:
	- Version mismatches between `tensorflow` and `keras` can cause import errors. Prefer using the integrated `tensorflow.keras` when possible. The code currently uses both `tensorflow.keras` (in `main.py`) and `keras` (in other files). If you hit errors, install a matching combination (for example, TensorFlow 2.x which includes `tf.keras`).

- Camera not opening on Linux:
	- In `main.py`, the code uses `cv2.VideoCapture(0)`. Try other indices (1, 2, ...). If using an IP camera or video file, pass the device path instead of `0`.

## Notes about code
- The recognition CNN expects 28x28 grayscale images normalized to [0,1]. The code centralizes digits using center-of-mass shifting before prediction.
- The solver writes only digits that were originally empty in the detected board.

## License
This repository does not contain an explicit license file. If you want to reuse this code, consider adding an appropriate license such as MIT or Apache-2.0.

## Acknowledgements
- Parts of the digit recognition and preprocessing were inspired by common MNIST examples and community tutorials.
- The project author documented a demo video referenced in the code comments (link in `RealTimeSudokuSolver.py`).

---

If you want, I can:
- Add a short script that runs on a saved video file instead of a webcam.
- Add a small system test that checks model loading and a smoke-run of the main loop (without camera).

Happy hacking!


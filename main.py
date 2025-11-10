# This is the entry point. Run this file!
# You don't need to run digitRecognition.py to train the Convolutional Neural Network (CNN).
# I have trained the CNN on my computer and saved the architecture in digitRecognition.h5

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import RealTimeSudokuSolver

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Load and set up Camera
cap = cv2.VideoCapture(0)  # Removed CAP_DSHOW for better Linux compatibility
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)    # HD Camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Loading model (Load weights and configuration separately to speed up model.predict)
input_shape = (28, 28, 1)
num_classes = 9

model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Load weights from pre-trained model. This model is trained in digitRecognition.py
model.load_weights("digitRecognition.h5")   

# Let's turn on webcam
old_sudoku = None
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        sudoku_frame = RealTimeSudokuSolver.recognize_and_solve_sudoku(frame, model, old_sudoku) 
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600) # Print the 'solved' image
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
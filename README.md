# Plant Disease Detection Model

This repository contains the code and resources for the Plant Disease Detection Model.
The model is designed to identify various plant diseases from images of plant leaves.

## Overview

The Plant Disease Detection Model uses a Convolutional Neural Network (CNN) to classify images of plant leaves into different disease categories.

## Model Architecture

The model is built using TensorFlow and Keras with the following architecture:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Results
**The model achieves an accuracy of 95% on the validation set. Below are the training accuracy and loss curves.**


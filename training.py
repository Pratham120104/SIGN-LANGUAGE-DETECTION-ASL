import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Mount Google Drive (for Colab use)
drive.mount('/content/drive')

# Dataset path
data_dir = '/content/drive/MyDrive/asl_dataset'
img_height, img_width = 64, 64

# Data preprocessing & augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir, target_size=(img_height, img_width),
    batch_size=32, class_mode='categorical', subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir, target_size=(img_height, img_width),
    batch_size=32, class_mode='categorical', subset='validation'
)

# CNN model
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save('models/asl_model.h5')

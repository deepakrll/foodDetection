!pip install tensorflow opencv-python-headless
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

IMG_SIZE = 224
BATCH_SIZE = 4
DATASET_PATH = '/content/drive/MyDrive/FoodSpoilageDataset'
TEMP_TRAIN_DIR = '/content/train_only'

# Clean and recreate train_only/fresh and train_only/spoiled
if os.path.exists(TEMP_TRAIN_DIR):
    shutil.rmtree(TEMP_TRAIN_DIR)
os.makedirs(os.path.join(TEMP_TRAIN_DIR, 'fresh'), exist_ok=True)
os.makedirs(os.path.join(TEMP_TRAIN_DIR, 'spoiled'), exist_ok=True)

# Copy only image files
for cls in ['fresh', 'spoiled']:
    src = os.path.join(DATASET_PATH, cls)
    dst = os.path.join(TEMP_TRAIN_DIR, cls)
    for fname in os.listdir(src):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(src, fname), dst)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.3  # Slightly more for training due to small size
)

train_data = datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Load pretrained base
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # Freeze the base model

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

import numpy as np
import cv2
import matplotlib.pyplot as plt

TEST_PATH = os.path.join(DATASET_PATH, 'test')
class_names = list(train_data.class_indices.keys())

def predict_test_images():
    for fname in os.listdir(TEST_PATH):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(TEST_PATH, fname)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            label = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100

            plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            plt.title(f"{fname}\nPredicted: {label} ({confidence:.2f}%)")
            plt.axis('off')
            plt.show()

predict_test_images()

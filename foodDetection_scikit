!pip install opencv-python-headless scikit-learn

from google.colab import drive
drive.mount('/content/drive')

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 100
BASE_PATH = '/content/drive/MyDrive/FoodSpoilageDataset'

# Load and preprocess images
def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flat = img_gray.flatten()
            images.append(img_flat)
            labels.append(label)
    return images, labels

# Load data
fresh_images, fresh_labels = load_images_from_folder(os.path.join(BASE_PATH, 'fresh'), 'fresh')
spoiled_images, spoiled_labels = load_images_from_folder(os.path.join(BASE_PATH, 'spoiled'), 'spoiled')

# Combine data
X = np.array(fresh_images + spoiled_images)
y = np.array(fresh_labels + spoiled_labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train the model
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print safe classification report
print("Classification Report:\n", classification_report(
    y_test, y_pred,
    labels=le.transform(le.classes_),
    target_names=le.classes_
))

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flat = img_gray.flatten().reshape(1, -1)
    prediction = model.predict(img_flat)
    label = le.inverse_transform(prediction)[0]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()

# Example usage
predict_image('/content/drive/MyDrive/FoodSpoilageDataset/test/test3.jpg')

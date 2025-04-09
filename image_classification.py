# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:59:28 2024

@author: stimp
"""

#importing libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#defining constants
IMG_SIZE = 224
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 10
NUM_IMAGES_TO_SHOW = 5  #number of images to show in the grid

#loading and preprocessing data
def load_data(data_dir):
    images = []
    labels = []
    class_labels = os.listdir(data_dir)
    for label, fruit in enumerate(class_labels):
        fruit_dir = os.path.join(data_dir, fruit)
        for img_name in os.listdir(fruit_dir):
            img = cv2.imread(os.path.join(fruit_dir, img_name))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

#importing dataset
data_dir = "C:/Users/stimp/OneDrive/Desktop/Flame/OPSM325/fruit project/dataset"
images, labels = load_data(data_dir)

#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#preprocessing images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#one-hot encoding labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#loading VGG16 model (imagenet)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

#freezing convolutional layers
for layer in vgg_model.layers:
    layer.trainable = False

#creating model
model = Sequential([
    vgg_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

#compiling model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator()

train_datagen.fit(X_train)
test_datagen.fit(X_test)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

#training the model
model.fit(train_generator, steps_per_epoch=len(X_train) // BATCH_SIZE, epochs=EPOCHS, validation_data=test_generator, validation_steps=len(X_test) // BATCH_SIZE)

#evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

#function to predict a single image input
def predict_single_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

#input path to image
image_path = "C:/Users/stimp/OneDrive/Desktop/Flame/OPSM325/fruit project/New folder/dragonfruit_example.jpg"

#predicting the class of the image
predicted_class = predict_single_image(image_path, model)

#mapping prediction to label
class_labels = os.listdir(data_dir)
fruit_name = class_labels[predicted_class]

print("Predicted class:", fruit_name)

#function to display grid of images
def display_images_grid(images, title):
    num_images = images.shape[0]
    rows = int(np.ceil(num_images / NUM_IMAGES_TO_SHOW))
    cols = min(num_images, NUM_IMAGES_TO_SHOW)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()

#train grid
display_images_grid(X_train[:NUM_IMAGES_TO_SHOW], "Train Images")

#test grid
display_images_grid(X_test[:NUM_IMAGES_TO_SHOW], "Test Images")

import keras
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib

import tensorflow as tf
from keras.src.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Rescaling, Flatten, Dense
from tensorflow.keras.models import Sequential


dataset_dir = pathlib.Path('/Users/innate/Desktop/flower_photos')
batch_size = 32
img_width, img_height = 180, 180

image_count = len(list(dataset_dir.glob("*/*.jpg")))
print(f"Total count images: {image_count}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create model
num_classes = len(class_names)
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Rescaling(1. / 255),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.2),

    # далі всі шари Conv2D та MaxPooling2D
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # регуляризация
    layers.Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# print(model.summary)
model.summary()

epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# visualize training and validation results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Збереження всієї моделі
model.save( '/Users/innate/Desktop/flowersModel.keras')


# load image
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#
# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)
#
# # make predictions
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
#
# # print inference result
# print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
#     class_names[np.argmax(score)],
#     100 * np.max(score)))
#
# # show the image itself
# img.show()

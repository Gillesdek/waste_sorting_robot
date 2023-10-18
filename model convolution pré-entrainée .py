import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


## import des données de test

batch_size = 32
img_height = 384
img_width = 512


data_dir = "/Users/gillesdeknache/Documents/TIPE/Base de données"

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

## récuperation du model pré-entrainée


model = tf.keras.models.load_model('model2.h5')


## prediction sur une image

path = "/Users/gillesdeknache/Desktop/bouteille.jpg"
img = keras.preprocessing.image.load_img(
path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print("l'image appartient à la classe "+str(class_names[np.argmax(score)])+" avec une précision de "+  str(np.round(np.max(score)*100))+"%")
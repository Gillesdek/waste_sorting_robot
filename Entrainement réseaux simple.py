import tensorflow as tf
from tensorflow import keras
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

os.chdir('/Users/gillesdeknache/Documents/TIPE/dataset-resized')


### definition des classes
class_names =["cardboard",
                "glass",
                "metal",
                "paper",
                "plastic"]


## Chargement de la base de données


train_images=np.zeros((1500,384,512))
train_labels=np.zeros(1500)
for i in range(0,300):
    img1 = cv2.imread("cardboard/cardboard"+str(i+1)+".jpg")
    img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    train_images[i]=img1
    train_labels[i]=0

for i in range (0,300):
    img2 = cv2.imread("glass/glass"+str(i+1)+".jpg")
    img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    train_images[i+300]=img2
    train_labels[i+300]=1

for i in range (0,300):
    img3 = cv2.imread("metal/metal"+str(i+1)+".jpg")
    img3= cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    train_images[i+600]=img3
    train_labels[i+600]=2

for i in range(0,300):
    img4 = cv2.imread("paper/paper"+str(i+1)+".jpg")
    img4= cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    train_images[i+900]=img4
    train_labels[i+900]=3

for i in range (0,300):
    img5 = cv2.imread("plastic/plastic"+str(i+1)+".jpg")
    img5= cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    train_images[i+1200]=img5
    train_labels[i+1200]=4


test_images=np.zeros((500,384,512))
test_labels=np.zeros(500)

for i in range(0,100):
    img1 = cv2.imread("cardboard/cardboard"+str(300+i+1)+".jpg")
    img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    test_images[i]=img1
    test_labels[i]=0

for i in range (0,100):
    img2 = cv2.imread("glass/glass"+str(300+i+1)+".jpg")
    img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    test_images[i+100]=img2
    test_labels[i+100]=1

for i in range (0,100):
    img3 = cv2.imread("metal/metal"+str(300+i+1)+".jpg")
    img3= cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    test_images[i+200]=img3
    test_labels[i+200]=2

for i in range(0,100):
    img4 = cv2.imread("paper/paper"+str(300+i+1)+".jpg")
    img4= cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    test_images[i+300]=img4
    test_labels[i+300]=3

for i in range (0,100):
    img5 = cv2.imread("plastic/plastic"+str(300+i+1)+".jpg")
    img5= cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    test_images[i+400]=img5
    test_labels[i+400]=4





##Definition du modèle



model=keras.Sequential([keras.layers.Flatten(input_shape=(384,512)),
                        keras.layers.Dense(128,activation=tf.nn.relu),
                        keras.layers.Dense(5,activation=tf.nn.softmax)])




## Entrainement


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
#model.fit(train_images,train_labels,epochs=5)


## Résultat


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions)


#image=cv2.imread("cardboard/cardboard302.jpg")
#image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#img = (np.expand_dims(image,0))

#predictions_single =model.predict(img)
#print(np.argmax(predictions_single))

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



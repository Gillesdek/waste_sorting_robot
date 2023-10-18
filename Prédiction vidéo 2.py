import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
img_height = 384
img_width = 512

cap = cv2.VideoCapture(0)

# chargement du model
model = tf.keras.models.load_model('model3.h5')

class_names = ['cardboard', 'plastic']

while(True):
    # enregister image par image
    ret, frame = cap.read()

    frame = cv2.resize(frame, (img_width,img_height))


    img_array = keras.preprocessing.image.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)

    # prediction sur chaque image
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    print("l'image appartient à la classe "+str(class_names[np.argmax(score)])+" avec une précision de "+  str(np.round(np.max(score)*100))+"%")


    # affichage
    cv2.imshow('frame',frame)

    #quitter le programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
import keras
import cv2
import os
import tensorflow as tf

model = keras.models.load_model('bain_V0.2.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('grey_0.1.tflite', 'wb') as f:
  f.write(tflite_model)



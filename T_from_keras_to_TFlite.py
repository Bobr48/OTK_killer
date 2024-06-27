import keras
import cv2
import os
import tensorflow as tf

M_ch_name = 'Models/testCH.bain_V0.5.2.keras'
M_TFlite_name = 'Models/grey_0.3.2.1.tflite'

model = keras.models.load_model(M_ch_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(M_TFlite_name, 'wb') as f:
  f.write(tflite_model)


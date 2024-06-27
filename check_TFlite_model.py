import numpy as np
import tensorflow as tf
import cv2
import os

kadr = 2
M_TFlite_name = 'Models/grey_0.3.2.1.tflite'


#expenddims
interpreter = tf.lite.Interpreter(model_path=M_TFlite_name)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()
l, names = [], []
c, c1=0, 0

if kadr==1:
    for file in os.listdir('foto_bad_test_1/'):
        c+=1
        img = cv2.imread('foto_bad_test_1/'+ str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32([img / 255])
        img = np.expand_dims(img, axis=-1)
        l.append(img)
        names.append(file)
if kadr==2:
    for file in os.listdir('foto_bad_test_2/'):
        c+=1
        img = cv2.imread('foto_bad_test_2/'+ str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32([img / 255])
        img = np.expand_dims(img, axis=-1)
        l.append(img)
        names.append(file)

for i in range(len(l)):
    interpreter.set_tensor(input_details[0]['index'], l[i])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = output_data[0][0] > 0.5
    if not result:
        c1+=1
    print(result, names[i])

res_bad = c1/len(os.listdir('foto_bad_test_2/'))*100

l, names = [], []
c, c1=0, 0

if kadr==1:
    for file in os.listdir('foto_ok_test_1/'):
        c+=1
        img = cv2.imread('foto_ok_test_1/'+ str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32([img / 255])
        img = np.expand_dims(img, axis=-1)
        l.append(img)
        names.append(file)
if kadr==2:
    for file in os.listdir('foto_ok_test_2/'):
        c+=1
        img = cv2.imread('foto_ok_test_2/'+ str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32([img / 255])
        img = np.expand_dims(img, axis=-1)
        l.append(img)
        names.append(file)

for i in range(len(l)):
    interpreter.set_tensor(input_details[0]['index'], l[i])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = output_data[0][0] > 0.5
    if result:
        c1 += 1
    print(result, names[i])

res_ok = c1 / len(os.listdir('foto_bad_test_2/')) * 100
print('res_bad = ', res_bad)
print('res_ok = ', res_ok)
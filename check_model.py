from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2
import os
names=[]

M_name = 'Models/testbrain_V0.6.1.keras'
M_ch_name = 'Models/testCH.bain_V0.6.1.keras'

####################################################
data = []
for i in os.listdir('foto_ok_1/'):
    img = cv2.imread('foto_ok_1/' + str(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append([img, [1, 0]])

np.random.shuffle(data)
data = data[:len(os.listdir('foto_bad_1/'))]
print(len(data))
for i in os.listdir('foto_bad_1/'):
    img = cv2.imread('foto_bad_1/' + str(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append([img, [0, 1]])
print(len(data))
np.random.shuffle(data)

x_train, y_train = [], []
for i in data:
    x_train.append(i[0]/255)
    y_train.append(i[1])

x_train = np.array(x_train)
y_train = np.array(y_train)

##############################################
data_test=[]
for i in os.listdir('foto_ok_test_1/'):
    img = cv2.imread('foto_ok_test_1/' + str(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data_test.append([img, [1, 0]])
    names.append('ok_' + i)
print(len(data_test))
for i in os.listdir('foto_bad_test_1/'):
    img = cv2.imread('foto_bad_test_1/' + str(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data_test.append([img, [0, 1]])
    names.append('bad_' + i)
print(len(data_test))
#np.random.shuffle(data_test)
x_test, y_test = [], []
for i in data_test:
    x_test.append(i[0]/255)
    y_test.append(i[1])

x_test = np.array(x_test)
y_test = np.array(y_test)
####################################################

model = keras.models.load_model(M_name)
model.evaluate(x_test, y_test)
model.evaluate(x_train, y_train)
for i in range(1):
    n = 1
    x = np.expand_dims(x_test[i], axis=0)
    res = model.predict(x)

    print(names[i], res)
    print(np.argmax(res))

model = keras.models.load_model(M_ch_name)
model.evaluate(x_test, y_test)
model.evaluate(x_train, y_train)
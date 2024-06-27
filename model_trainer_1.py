from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2
import os

M_name = 'Models/testbrain_V0.6.1.keras'
M_ch_name = 'Models/testCH.bain_V0.6.1.keras'
names = []

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
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
#######################################

y_train_cat = keras.utils.to_categorical(y_train, 2)
y_test_cat = keras.utils.to_categorical(y_test, 2)

#model = keras.Sequential([Flatten(input_shape=(200,240,1))
#                             , Dense(1000, activation='relu'),Dropout(0.5), Dense(100, activation='relu'), Dense(2, activation='softmax')])

model = keras.Sequential([Conv2D(16,(3,3), padding='same', activation='relu', input_shape=(200,240,1)),
                          Conv2D(16,(3,3), padding='same', activation='relu'),
                          Conv2D(16,(3,3), padding='same', activation='relu'),
                          MaxPooling2D((2,2), strides=2),
                          Conv2D(32, (3, 3), padding='same', activation='relu'),
                          Conv2D(32, (3, 3), padding='same', activation='relu'),
                          MaxPooling2D((2,2), strides=2),
                          Conv2D(64, (3, 3), padding='same', activation='relu'),
                          Conv2D(64, (3, 3), padding='same', activation='relu'),
                          MaxPooling2D((2,2), strides=2),
                          Flatten(),
                          Dense(256 , activation='relu'),Dropout(0.25),
                          #Dense(50, activation='relu'),
                          Dense(2, activation='softmax')])

# model = keras.Sequential([Conv2D(32,(5,5), padding='same', activation='relu', input_shape=(200,240,1)),
#                          MaxPooling2D((3,3), strides=2),Dropout(0.25),
#                          Conv2D(64, (3, 3), padding='same', activation='relu'),
#                          Conv2D(64, (3, 3), padding='same', activation='relu'),
#                          MaxPooling2D((3,3), strides=2),Dropout(0.25),
#
#                          Flatten(),
#                          Dense(1024 , activation='relu'),Dropout(0.5),
#                          Dense(100, activation='relu'), Dropout(0.25),
#                          Dense(2, activation='softmax')])



print(model.summary())

model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
callback = keras.callbacks.ModelCheckpoint(filepath=M_ch_name, save_best_only=True, mode='min', monitor="val_loss", verbose=1)

model.fit(x_train, y_train, batch_size = 32, epochs = 7, callbacks = [callback],  validation_split = 0.2)

model.save(M_name)
##############################################################
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)
for i in range(1):
    n = 1
    x = np.expand_dims(x_test[i], axis=0)
    res = model.predict(x)

    print(names[i], res)
    print(np.argmax(res))
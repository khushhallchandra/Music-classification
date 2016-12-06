from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import load_data

batch_size = 5
nb_classes = 10
nb_epoch = 10

# input image dimensions
# img_rows, img_cols = 128, 1290
img_rows, img_cols = 96, 1366
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Convolution2D(2*nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Convolution2D(2*nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Convolution2D(4*nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

for e in range(9):
    print("epoch %d" % e)
    X_train, Y_train = load_data1.data(e)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=nb_epoch)

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
X_test, Y_test = load_data1.data(10)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

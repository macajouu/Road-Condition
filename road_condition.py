##Imports 


import numpy as np
np.random.seed(1337)  # for reproducibility

import os
os.environ["KERAS_BACKEND"] = "theano"
os.environ["image_dim_ordering"] = "tf"

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda0, floatX=float32, exception_verbosity=high" 

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import cv2
import matplotlib.pyplot as plt

##Variable initializations


batch_size = 64
nb_classes = 3
nb_epoch = 40

# input image dimensions
img_rows, img_cols = 245, 640

# size of pooling area for max pooling
pool_size = (2, 2)

## Input - train and test datasets

#    ---------DATASET---------

test_src_directory = r'dataset'
validation_src_directory = r'dataset'

X_test = np.memmap(os.path.join(test_src_directory, 'X_test_images.h5'), dtype='float32', mode='r', shape=(100, 245, 640))
Y_test = np.memmap(os.path.join(test_src_directory, 'Y_test_labels.h5'), dtype='float32', mode='r', shape=(100, 1))

X_validation = np.memmap(os.path.join(validation_src_directory, 'X_test_images.h5'), dtype='float32', mode='r', shape=(100, 245, 640))
Y_validation = np.memmap(os.path.join(validation_src_directory, 'Y_test_labels.h5'), dtype='float32', mode='r', shape=(100, 1))


## Ordering, reshaping, categorical

if K.image_dim_ordering() == 'th':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
Y_test = np_utils.to_categorical(Y_test, 3)
Y_validation = np_utils.to_categorical(Y_validation, 3)

# print(Y_train[:10])
print(Y_validation.shape)

## Assemble the model

print('Start assembling model')

model = Sequential()
 
model.add(Conv2D(12, (7, 7), strides=(2, 2), input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

print('1st OUTPUT SHAPE: ' + str(model.output_shape))

model.add(Conv2D(24, (5,5), strides=(2, 2), kernel_regularizer = regularizers.l2(0.01)))
model.add(Activation('relu'))

model.add(Conv2D(24, (5,5), strides=(1, 1), kernel_regularizer = regularizers.l2(0.01)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


print('2nd OUTPUT SHAPE: ' + str(model.output_shape))

model.add(Conv2D(48, (3, 3), kernel_regularizer = regularizers.l2(0.01)))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3), kernel_regularizer = regularizers.l2(0.01)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
 
print('3rd OUTPUT SHAPE: ' + str(model.output_shape))
 
model.add(Flatten())

print('Flatten: ' + str(model.output_shape))

model.add(Dense(256, activation='relu', kernel_regularizer = regularizers.l2(0.01)))

print('Fully connected: ' + str(model.output_shape))

model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

print('Assembled model')


## Compile the model


print('Start compiling model')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Compiled model')

## Load and manipulate weights
model.load_weights('models/weights.hdf5')

## Callback to save weights

# callbacks = []
# callbacks.append(ModelCheckpoint('models/weights.hdf5', save_best_only=True))

## Train the model

# print('Start fitting model')

# model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
#           verbose=1, validation_data=(X_validation, Y_validation), callbacks=callbacks)

# print('Model fit')


## Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1);
print('Test loss:', score[0]);
print('Test accuracy:', score[1]);


## View an image 

index = 50

img_sample = X_validation[index]
#     img_sample = X_test[index]
#     img_sample = X_train[index]

sh = img_sample.shape
img = img_sample.reshape(sh[0], sh[1])

plt.figure(figsize=(24,10))
plt.imshow(img, cmap='gray', aspect='auto');
# plt.colorbar()

score = model.predict(img_sample.reshape(1, sh[0], sh[1], sh[2]))
score = score[0] * 100
score = [round(x,2) for x in score]
categs = ['Dry', 'Wet', 'Snow']

#     label_categ = Y_test[index]
label_categ = Y_validation[index]

label = label_categ.argmax()
print('Expected: ' + categs[label] + '\n')

for i in range(len(score)):
	print(categs[i] + ' - ' + str(score[i]) + ' %')



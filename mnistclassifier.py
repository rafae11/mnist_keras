'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import matplotlib.pyplot as plt
import pandas as pd

img_rows, img_cols = 28, 28
batch_size = 128
nb_classes = 10
nb_epoch = 20

input_shape = (1, 28, 28) # image shape

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# read test data from kaggle
data = pd.read_csv('test.csv')

target = open('mnist_submission.csv', 'w')
target.write('ImageId,Label\n')
for i in range(0,len(data)):
    input = data.ix[i][:].values
    prediction = model.predict(input.reshape(-1,28,28,1))
    label = np.round(max(prediction))
    index, value = max(enumerate(label), key=lambda v: v[1])
    #print('predicted value is %s' %(index))
    target.write("%s,%s\n" %(i+1,index))
target.close()
      
print('session complete')
"""
plt.imshow(X_train[idx].reshape(28,28))
plt.title(Y_train[idx])
plt.show()
"""
K.clear_session()
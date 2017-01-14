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
from keras.layers.convolutional import ZeroPadding2D
from sklearn.cross_validation import train_test_split
import pandas as pd
from preprocessing import preprocess_image
import keras.utils.np_utils as kutils

img_rows, img_cols = 28, 28
img_height , img_width = 28 , 28
batch_size = 128
nb_classes = 10
nb_epoch = 10

train = pd.read_csv("train.csv").values
test = pd.read_csv("test.csv").values

input_shape = (1, 28, 28) # image shape

X_train = train[:, 1:].reshape(train.shape[0], img_height, img_width,1)
#X_train = X_train.astype(float)
#X_train /= 255.0
#X_train = np.array([preprocess_image(x[0]).reshape(img_height, img_width,1) for x in X_train])

print (X_train.shape)

Y_train = kutils.to_categorical(train[:, 0])
nb_classes = Y_train.shape[1]

# Split the training data into training and validation data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

print (X_train.shape)
print (Y_test.shape)
print (Y_train.shape)
print (Y_test.shape)

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

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

K.clear_session()
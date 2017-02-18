"""

Filename: tensor_flow_sa.py
Author: Steven
Date Last Modified: 2/2/2017
Email: bouwkast@mail.gvsu.edu

"""
import datetime
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb


# set parameters:
max_features = 50000
maxlen = 400
batch_size = 64
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 30

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# print(X_train)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/' + now.strftime('%Y.%m.%d %H.%M'))
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          callbacks=[EarlyStopping(patience=5),
                     tensorboard],
          batch_size=batch_size,
          verbose=2,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))









# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:39:16 2022

@author: Utente
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt



##### Load input data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

n_clusters=10
dims=[x_train.shape[-1], 500, 500, 2000, n_clusters]


def autoencoder(dims, act='relu', init='glorot_uniform'):
    
    n_stacks = len(dims) - 1
    
    input_img = Input(shape=(dims[0],),name='input')
    h= input_img
    
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer, features are extracted from here
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  

    y = h
     # internal layers in decoder 
    for i in range(n_stacks-1, 0, -1):
         y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
    
    #output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)# decoded representation of code 
    
    return Model(inputs=input_img, outputs=y, name='AE'), Model(inputs=input_img, outputs=h, name='encoder')
    
    
autoencoder, encoder = autoencoder(dims, init='glorot_uniform')
# Model which take input image and shows decoded images



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



autoencoder.fit(x_train, x_train,
                epochs=15,
                batch_size=256,
                validation_data=(x_test, x_test))

encoded_img = encoder.predict(x_test)
autoencoded_img = autoencoder.predict(x_test)


plt.figure(figsize=(20, 4))
for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(autoencoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



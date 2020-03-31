# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aISRGIwYaDXuzGaOVef2zv1PKl1dJLZB
"""

# Commented out IPython magic to ensure Python compatibility.
# SUDHEER NADELLA          U93511802
# Run this program in google colab if you face any problem executing this in your remote environment. 
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing
# %tensorflow_version 1.x
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model,Sequential

# Our Test set values
Test= np.array([
     [1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]])

# Defining number of nodes in Input, Hidden and Output layers.
input_layer_nodes = 4
hidden_layer_nodes = 2
output_layer_nodes = 4

# Defining Input, hidden and output layers.
input_layer = Input(shape=(input_layer_nodes,))
hidden_layer = Dense(hidden_layer_nodes, activation='relu')(input_layer)
output_layer = Dense(output_layer_nodes, activation='softmax')(hidden_layer)

# Defining Autoencoder
autoencoder = Model(input=input_layer, output=output_layer)
autoencoder.compile(optimizer='nadam', loss='mse', metrics=['acc'])
history= autoencoder.fit(Test, Test, batch_size=64, nb_epoch=1500)
# evaluated_Test = autoencoder.predict(Test)

# Printing the evaluated values 
# print(evaluated_Test)
# Printing the rounded evaluated values
# print('rounded values')
# evaluated_Test = evaluated_Test.round()
# print(evaluated_Test)

#####  Project extended    #######
#####  Extract Features from auto encoder  #########


# Defining number of nodes in Input, Hidden and Output layers.
new_hidden_layer_nodes = 10
new_output_layer_nodes = 4

new_model = Sequential()
new_model.add(Dense(2, input_dim = 4, activation = 'tanh'))
new_model.set_weights(autoencoder.layers[0].get_weights())
new_model.compile(optimizer='nadam', loss='mse')

output = new_model.predict(Test)

# Defining New Input, hidden and output layers.
new_input_layer = Input(shape=(2,))
new_hidden_layer = Dense(new_hidden_layer_nodes, activation='relu')(new_input_layer)
new_output_layer = Dense(new_output_layer_nodes, activation='softmax')(new_hidden_layer)

# New auto encoder
new_autoencoder = Model(input=new_input_layer, output=new_output_layer)
new_autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
history1= new_autoencoder.fit(output, Test, batch_size=4, nb_epoch=1500)
# print(new_autoencoder.predict(output))
new_evaluated_Test = new_autoencoder.predict(output)

print(new_evaluated_Test)
# Rounded values
new_evaluated_Test = new_evaluated_Test.round()
print(new_evaluated_Test)


####  Outputs  #######
# [[9.5610565e-01 9.9300560e-06 1.8584561e-02 2.5299812e-02]
#  [2.1615343e-03 8.3028489e-01 1.0630469e-02 1.5692310e-01]
#  [2.5279276e-02 2.6762007e-02 9.3600929e-01 1.1949487e-02]
#  [6.5840883e-03 1.6113734e-01 1.2855931e-02 8.1942260e-01]]
## Rounded Values ##
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
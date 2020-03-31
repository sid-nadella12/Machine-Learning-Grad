# SUDHEER NADELLA          U93511802
# Run this program in google colab if you face any problem executing this in your remote environment. 
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing
from keras.layers import Input, Dense
from keras.models import Model

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
history= autoencoder.fit(Test, Test, batch_size=64, nb_epoch=1000)
evaluated_Test = autoencoder.predict(Test)

# Printing the evaluated values 
print(evaluated_Test)

# Printing the rounded evaluated values
print('rounded values')
evaluated_Test = evaluated_Test.round()
print(evaluated_Test)


#####  Project extended    #######


# Defining number of nodes in Input, Hidden and Output layers.
new_input_layer_nodes = 2
new_hidden_layer_nodes = 2
new_output_layer_nodes = 4

# extended project
new_input_layer = Input()
new_hidden_layer = Dense(new_hidden_layer_nodes, activation = 'relu')(hidden_layer)
new_output_layer = Dense(new_output_layer_nodes, activation='softmax')(new_hidden_layer)

# Defining New Autoencoder
new_autoencoder = Model(input=hidden_layer, output=new_output_layer)
new_autoencoder.compile(optimizer='nadam', loss='mse', metrics=['acc'])
history= new_autoencoder.fit(Test, Test, batch_size=64, nb_epoch=1000)
new_evaluated_Test = new_autoencoder.predict(Test)





# OUTPUT

#[[9.1763788e-01 2.1301989e-02 6.0750879e-02 3.0911807e-04]
# [4.8192892e-02 8.9931232e-01 1.6170779e-02 3.6324024e-02]
# [4.1554432e-02 1.6701390e-03 9.2474866e-01 3.2026742e-02]
# [4.5505751e-04 2.2492317e-02 4.4143021e-02 9.3290967e-01]]
#rounded values
#[[1. 0. 0. 0.]
# [0. 1. 0. 0.]
# [0. 0. 1. 0.]
# [0. 0. 0. 1.]]
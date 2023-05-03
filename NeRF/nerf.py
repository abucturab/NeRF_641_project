# import the necessary packages
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Input
from tensorflow.keras import Model
import tensorflow as tf

""" Build the MLP for NeRF """
def get_model(lxyz, lDir, batchSize, denseUnits, skipLayer):
	# build input layer for rays
	rayInput = Input(shape=(None, None, None, 2 * 3 * lxyz + 3), batch_size=batchSize)
	
	# build input layer for direction of the rays
	dirInput = Input(shape=(None, None, None, 2 * 3 * lDir + 3), batch_size=batchSize)
	
	# creating an input for the MLP
	x = rayInput
	for i in range(8):
		# build a dense layer
		x = Dense(units=denseUnits, activation="relu")(x)
		# check if we have to include residual connection
		if i % skipLayer == 0 and i > 0:
			# inject the residual connection
			x = concatenate([x, rayInput], axis=-1)
	
	# get the sigma value
	sigma = Dense(units=1, activation="relu")(x)
	# create the feature vector
	feature = Dense(units=denseUnits)(x)
	# concatenate the feature vector with the direction input and put
	# it through a dense layer
	feature = concatenate([feature, dirInput], axis=-1)
	x = Dense(units=denseUnits//2, activation="relu")(feature)
	# get the rgb value
	rgb = Dense(units=3, activation="sigmoid")(x)
	# create the nerf model
	nerfModel = Model(inputs=[rayInput, dirInput], outputs=[rgb, sigma])
	
	# return the nerf model
	return nerfModel




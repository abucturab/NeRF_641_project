# import the necessary packages
import tensorflow as tf
def encoder_fn(p, L):
	# build the list of positional encodings
	gamma = [p]
	# iterate over the number of dimensions in time
	for i in range(L):
		# insert sine and cosine of the product of current dimension
		# and the position vector
		gamma.append(tf.sin((2.0 ** i) * p))
		gamma.append(tf.cos((2.0 ** i) * p))
	
	# concatenate the positional encodings into a positional vector
	gamma = tf.concat(gamma, axis=-1)
	# return the positional encoding vector
	return gamma
import tensorflow as tf
import math
from . import config
import numpy as np
def sample_pdf(tValsMid, weights, nF):
	# add a small value to the weights to prevent it from nan
	weights += 1e-5
	# normalize the weights to get the pdf
	pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
	# from pdf to cdf transformation
	cdf = tf.cumsum(pdf, axis=-1)
	# start the cdf with 0s
	cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)
	# get the sample points
	uShape = [config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, nF]
	u = tf.random.uniform(shape=uShape)
	# get the indices of the points of u when u is inserted into cdf in a
	# sorted manner
	indices = tf.searchsorted(cdf, u, side="right")
	# define the boundaries
	below = tf.maximum(0, indices-1)
	above = tf.minimum(cdf.shape[-1]-1, indices)
	indicesG = tf.stack([below, above], axis=-1)
	
	# gather the cdf according to the indices
	cdfG = tf.gather(cdf, indicesG, axis=-1,
		batch_dims=len(indicesG.shape)-2)
	
	# gather the tVals according to the indices
	tValsMidG = tf.gather(tValsMid, indicesG, axis=-1,
		batch_dims=len(indicesG.shape)-2)
	# create the samples by inverting the cdf
	denom = cdfG[..., 1] - cdfG[..., 0]
	denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
	t = (u - cdfG[..., 0]) / denom
	samples = (tValsMidG[..., 0] + t * 
		(tValsMidG[..., 1] - tValsMidG[..., 0]))
	
	# return the samples
	return samples

""" Volume Rendering Routine """

def render_image_depth(rgb, sigma, tVals):
	# squeeze the last dimension of sigma
	sigma = sigma[..., 0]
	# calculate the delta between adjacent tVals
	delta = tVals[..., 1:] - tVals[..., :-1]
	deltaShape = [config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1]
	delta = tf.concat([delta, tf.broadcast_to([1e10], shape=deltaShape)], axis=-1)
	# calculate alpha from sigma and delta values
	alpha = 1.0 - tf.exp(-sigma * delta)
	# calculate the exponential term for easier calculations
	expTerm = 1.0 - alpha
	epsilon = 1e-10
	# calculate the transmittance and weights of the ray points
	transmittance = tf.math.cumprod(expTerm + epsilon, axis=-1, exclusive=True)
	weights = alpha * transmittance
	
	# build the image and depth map from the points of the rays
	image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
	depth = tf.reduce_sum(weights * tVals, axis=-1)
	
	# return rgb, depth map and weights
	return (image, depth, weights)


def get_focal_from_fov(fieldOfView, width):
	focal_length = (width / 2) / math.tan(fieldOfView / 2)
	return focal_length

def rotation_matrix(theta, phi):
	theta_rad = np.deg2rad(theta)
	phi_rad = np.deg2rad(phi)

	rz_theta = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
							[np.sin(theta_rad),  np.cos(theta_rad), 0],
							[0,                          0,         1]])

	ry_phi = np.array([[ np.cos(phi_rad), 0, np.sin(phi_rad)],
						[       0,         1,       0        ],
						[-np.sin(phi_rad), 0, np.cos(phi_rad)]])

	rotation = rz_theta @ ry_phi

	return rotation

def translation_matrix(distance):
	translation = np.array([[1, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, -distance],
							[0, 0, 0, 1]])
	return translation

def pose_spherical(theta, phi, distance):
	rotation = rotation_matrix(theta, phi)
	translation = translation_matrix(distance)

	transformation = np.eye(4)
	transformation[:3, :3] = rotation
	transformation = transformation @ translation

	return transformation.astype(np.float32)
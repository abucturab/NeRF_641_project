# import the necessary packages
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg
from tensorflow.image import convert_image_dtype
from tensorflow.image import resize
from tensorflow import reshape
import tensorflow as tf
import json

""" Take the json Path and return the dictionary """
# Input: path
# Output: dictionary
def read_json(jsonPath):
	# open the json file
	with open(jsonPath, "r") as fp:
		# read the json data
		data = json.load(fp)
	
	# return the data
	return data

""" Read the json dictionary and return the image path and extrinsic matrix
    i.e., CameraToWorld coordinates transformation matrix """
def get_image_c2w(jsonData, datasetPath):
	# define a list to store the image paths
	imagePaths = []
	
	# define a list to store the camera2world matrices
	c2ws = []
	# iterate over each frame of the data
	for frame in jsonData["frames"]:
		# grab the image file name
		imagePath = frame["file_path"]
		imagePath = imagePath.replace(".", datasetPath)
		imagePaths.append(f"{imagePath}.png")
		# grab the camera2world matrix
		c2ws.append(frame["transform_matrix"])
	
	# return the image file names and the camera2world matrices
	return (imagePaths, c2ws)

""" Class which reads the images from a path and returns a tensor """
""" Used with tf.data.Dataset in the map call """
class GetImages():
	def __init__(self, imageWidth, imageHeight):
		# define the image width and height
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight
	def __call__(self, imagePath):
		# read the image file
		image = read_file(imagePath)
		# decode the image string
		image = decode_jpeg(image, 3)
		# convert the image dtype from uint8 to float32
		image = convert_image_dtype(image, dtype=tf.float32)
		# resize the image to the height and width in config
		image = resize(image, (self.imageWidth, self.imageHeight))
		image = reshape(image, (self.imageWidth, self.imageHeight, 3))
		# return the image
		return image
	
class GetRays:
	def __init__(self, focalLength, imageWidth, imageHeight, near, far, nC):
		# define the focal length, image width, and image height
		self.focalLength = focalLength
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight
		# define the near and far bounding values
		self.near = near
		self.far = far
		# define the number of samples for coarse model
		self.nC = nC
		
	def __call__(self, camera2world):
		# create a meshgrid of image dimensions
		(x, y) = tf.meshgrid(
			tf.range(self.imageWidth, dtype=tf.float32),
			tf.range(self.imageHeight, dtype=tf.float32),
			indexing="xy",
		)
		# define the camera coordinates
		xCamera = (x - self.imageWidth * 0.5) / self.focalLength
		yCamera = (y - self.imageHeight * 0.5) / self.focalLength
		# define the camera vector
		xCyCzC = tf.stack([xCamera, -yCamera, -tf.ones_like(x)],
			axis=-1)
		# slice the camera2world matrix to obtain the rotation and
		# translation matrix
		rotation = camera2world[:3, :3]
		translation = camera2world[:3, -1]
		
		# expand the camera coordinates to 
		xCyCzC = xCyCzC[..., None, :]
		
		# get the world coordinates
		xWyWzW = xCyCzC * rotation
		
		# calculate the direction vector of the ray
		rayD = tf.reduce_sum(xWyWzW, axis=-1)
		rayD = rayD / tf.norm(rayD, axis=-1, keepdims=True)
		# calculate the origin vector of the ray
		rayO = tf.broadcast_to(translation, tf.shape(rayD))
		# get the sample points from the ray
		tVals = tf.linspace(self.near, self.far, self.nC)
		noiseShape = list(rayO.shape[:-1]) + [self.nC]
		noise = (tf.random.uniform(shape=noiseShape) * (self.far - self.near) / self.nC)
		tVals = tVals + noise
		# return ray origin, direction, and the sample points
		return (rayO, rayD, tVals)
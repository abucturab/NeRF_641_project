# USAGE
# python train.py
# setting seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)
# import the necessary packages
from NeRF.data import read_json
from NeRF.data import get_image_c2w
from NeRF.data import GetImages
from NeRF.data import GetRays
from NeRF import config
from NeRF.utils import get_focal_from_fov, render_image_depth, sample_pdf
from NeRF.encoder import encoder_fn
from NeRF.nerf import get_model
from NeRF.nerf_trainer import Nerf_Trainer
from NeRF.train_monitor import get_train_monitor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os

# get the train validation and test data
print("[INFO] grabbing the data from json files...")
jsonTrainData = read_json(config.TRAIN_JSON)
jsonValData = read_json(config.VAL_JSON)
jsonTestData = read_json(config.TEST_JSON)

focalLength = get_focal_from_fov(
	fieldOfView=jsonTrainData["camera_angle_x"],
	width=config.IMAGE_WIDTH)

#focalLength = 525.0

# print the focal length of the camera
print(f"[INFO] focal length of the camera: {focalLength}...")

# get the train, validation, and test image paths and camera2world
# matrices
print("[INFO] grabbing the image paths and camera2world matrices...")
trainImagePaths, trainC2Ws = get_image_c2w(jsonData=jsonTrainData, datasetPath=config.DATASET_PATH)
valImagePaths, valC2Ws = get_image_c2w(jsonData=jsonValData, datasetPath=config.DATASET_PATH)
testImagePaths, testC2Ws = get_image_c2w(jsonData=jsonTestData, datasetPath=config.DATASET_PATH)

# instantiate a object of our class used to load images from disk
getImages = GetImages(imageHeight=config.IMAGE_HEIGHT, imageWidth=config.IMAGE_WIDTH)

# get the train, validation, and test image dataset
print("[INFO] building the image dataset pipeline...")
trainImageDs = (
	tf.data.Dataset.from_tensor_slices(trainImagePaths)
	.map(getImages, num_parallel_calls=config.AUTO)
)
valImageDs = (
	tf.data.Dataset.from_tensor_slices(valImagePaths)
	.map(getImages, num_parallel_calls=config.AUTO)
)
testImageDs = (
	tf.data.Dataset.from_tensor_slices(testImagePaths)
	.map(getImages, num_parallel_calls=config.AUTO)
)
# instantiate the GetRays object
getRays = GetRays(focalLength=focalLength, imageWidth=config.IMAGE_WIDTH, imageHeight=config.IMAGE_HEIGHT, near=config.NEAR, far=config.FAR, nC=config.N_C)
# get the train validation and test rays dataset
print("[INFO] building the rays dataset pipeline...")
trainRayDs = (
	tf.data.Dataset.from_tensor_slices(trainC2Ws)
	.map(getRays, num_parallel_calls=config.AUTO)
)
valRayDs = (
	tf.data.Dataset.from_tensor_slices(valC2Ws)
	.map(getRays, num_parallel_calls=config.AUTO)
)
testRayDs = (
	tf.data.Dataset.from_tensor_slices(testC2Ws)
	.map(getRays, num_parallel_calls=config.AUTO)
)

# zip the images and rays dataset together
trainDs = tf.data.Dataset.zip((trainRayDs, trainImageDs))
valDs = tf.data.Dataset.zip((valRayDs, valImageDs))
testDs = tf.data.Dataset.zip((testRayDs, testImageDs))

# build data input pipeline for train, val, and test datasets
trainDs = (
	trainDs
	.shuffle(config.BATCH_SIZE)
	.batch(config.BATCH_SIZE)
	.repeat()
	.prefetch(config.AUTO)
)
valDs = (
	valDs
	.shuffle(config.BATCH_SIZE)
	.batch(config.BATCH_SIZE)
	.repeat()
	.prefetch(config.AUTO)
)
testDs = (
	testDs
	.batch(config.BATCH_SIZE)
	.prefetch(config.AUTO)
)
# instantiate the coarse model
coarseModel = get_model(lxyz=config.L_XYZ, lDir=config.L_DIR,
	batchSize=config.BATCH_SIZE, denseUnits=config.DENSE_UNITS,
	skipLayer=config.SKIP_LAYER)
# instantiate the fine model
fineModel = get_model(lxyz=config.L_XYZ, lDir=config.L_DIR,
	batchSize=config.BATCH_SIZE, denseUnits=config.DENSE_UNITS,
	skipLayer=config.SKIP_LAYER)
# instantiate the nerf trainer model
nerfTrainerModel = Nerf_Trainer(coarseModel=coarseModel, fineModel=fineModel,
	lxyz=config.L_XYZ, lDir=config.L_DIR, encoderFn=encoder_fn,
	renderImageDepth=render_image_depth, samplePdf=sample_pdf,
	nF=config.N_F)
# compile the nerf trainer model with Adam optimizer and MSE loss
nerfTrainerModel.compile(optimizerCoarse=Adam(),optimizerFine=Adam(),
	lossFn=MeanSquaredError())

# check if the output image directory already exists, if it doesn't, then create it
if not os.path.exists(config.IMAGE_PATH):
	os.makedirs(config.IMAGE_PATH)
	
# get the train monitor callback
# TODO: trainDs doesn't exits so modify callback to not run test set
trainMonitorCallback = get_train_monitor(testDs=testDs, encoderFn=encoder_fn, lxyz=config.L_XYZ, lDir=config.L_DIR, imagePath=config.IMAGE_PATH)

# train the NeRF model
print("[INFO] training the nerf model...")
nerfTrainerModel.fit(trainDs, steps_per_epoch=config.STEPS_PER_EPOCH, validation_data=valDs, validation_steps=config.VALIDATION_STEPS, epochs=config.EPOCHS, callbacks=[trainMonitorCallback],)
# save the coarse and fine model
nerfTrainerModel.coarseModel.save(config.COARSE_PATH)
nerfTrainerModel.fineModel.save(config.FINE_PATH)
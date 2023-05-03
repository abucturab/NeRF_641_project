# import the necessary packages
import tensorflow as tf
from tensorflow.keras.metrics import Mean


class Nerf_Trainer(tf.keras.Model):
	def __init__(self, coarseModel, fineModel, lxyz, lDir, encoderFn, renderImageDepth, samplePdf, nF):
		super().__init__()
		# define the coarse model and fine model
		self.coarseModel = coarseModel
		self.fineModel = fineModel
		# define the dimensions for positional encoding for spatial
		# coordinates and direction
		self.lxyz = lxyz
		self.lDir = lDir
		# define the positional encoder
		self.encoderFn = encoderFn
		# define the volume rendering function
		self.renderImageDepth = renderImageDepth
		# define the hierarchical sampling function and the number of
		# samples for the fine model
		self.samplePdf = samplePdf
		self.nF = nF
		
	def compile(self, optimizerCoarse, optimizerFine, lossFn):
		super().compile()
		# define the optimizer for the coarse and fine model
		self.optimizerCoarse = optimizerCoarse
		self.optimizerFine = optimizerFine
		# define the photometric loss function
		self.lossFn = lossFn
		# define the loss and psnr tracker
		self.lossTracker = Mean(name="loss")
		self.psnrMetric = Mean(name="psnr")
		
	def train_step(self, inputs):
		
		# get the images and the rays
		(elements, images) = inputs
		(raysOriCoarse, raysDirCoarse, tValsCoarse) = elements
		
		# generate the coarse rays
		raysCoarse = (raysOriCoarse[..., None, :] + (raysDirCoarse[..., None, :] * tValsCoarse[..., None]))
		
		# positional encode the rays and dirs
		raysCoarse = self.encoderFn(raysCoarse, self.lxyz)
		dirCoarseShape = tf.shape(raysCoarse[..., :3])
		dirsCoarse = tf.broadcast_to(raysDirCoarse[..., None, :], shape=dirCoarseShape)
		dirsCoarse = self.encoderFn(dirsCoarse, self.lDir)
		 
		# keep track of our gradients
		with tf.GradientTape() as coarseTape:
			# compute the predictions from the coarse model
			(rgbCoarse, sigmaCoarse) = self.coarseModel([raysCoarse, dirsCoarse])
			
			# render the image from the predictions
			renderCoarse = self.renderImageDepth(rgb=rgbCoarse, sigma=sigmaCoarse, tVals=tValsCoarse)
			(imagesCoarse, _, weightsCoarse) = renderCoarse
			# compute the photometric loss
			lossCoarse = self.lossFn(images, imagesCoarse) #TODO Define the loss function 
			
		# compute the middle values of t vals
		tValsCoarseMid = (0.5 * (tValsCoarse[..., 1:] + tValsCoarse[..., :-1]))
		
		# apply hierarchical sampling and get the t vals for the finemodel
		tValsFine = self.samplePdf(tValsMid=tValsCoarseMid, weights=weightsCoarse, nF=self.nF)
		tValsFine = tf.sort(tf.concat([tValsCoarse, tValsFine], axis=-1), axis=-1)
		
        # build the fine rays and positional encode it
		raysFine = (raysOriCoarse[..., None, :] + (raysDirCoarse[..., None, :] * tValsFine[..., None]))
		raysFine = self.encoderFn(raysFine, self.lxyz)
		
		# build the fine directions and positional encode it
		dirsFineShape = tf.shape(raysFine[..., :3])
		dirsFine = tf.broadcast_to(raysDirCoarse[..., None, :], shape=dirsFineShape)
		dirsFine = self.encoderFn(dirsFine, self.lDir)
		
		# keep track of our gradients
		with tf.GradientTape() as fineTape:
			# compute the predictions from the fine model
			rgbFine, sigmaFine = self.fineModel([raysFine, dirsFine])
			
			# render the image from the predictions
			renderFine = self.renderImageDepth(rgb=rgbFine, sigma=sigmaFine, tVals=tValsFine)
			(imageFine, _, _) = renderFine
			# compute the photometric loss
			lossFine = self.lossFn(images, imageFine)

		# get the trainable variables from the coarse model and apply back propagation
		tvCoarse = self.coarseModel.trainable_variables
		gradsCoarse = coarseTape.gradient(lossCoarse, tvCoarse)
		self.optimizerCoarse.apply_gradients(zip(gradsCoarse, tvCoarse))
		
		# get the trainable variables from the fine model and apply back propagation
		tvFine = self.fineModel.trainable_variables
		gradsFine = fineTape.gradient(lossFine, tvFine)
		self.optimizerFine.apply_gradients(zip(gradsFine, tvFine))
		psnr = tf.image.psnr(images, imageFine, max_val=1.0)
		
        # compute the loss and psnr metrics
		self.lossTracker.update_state(lossFine)
		self.psnrMetric.update_state(psnr)
		# return the loss and psnr metrics
		return {"loss": self.lossTracker.result(), "psnr": self.psnrMetric.result()}
	
	def test_step(self, inputs):
		# get the images and the rays
		(elements, images) = inputs
		(raysOriCoarse, raysDirCoarse, tValsCoarse) = elements
		# generate the coarse rays
		raysCoarse = (raysOriCoarse[..., None, :] + 
			(raysDirCoarse[..., None, :] * tValsCoarse[..., None]))
		# positional encode the rays and dirs
		raysCoarse = self.encoderFn(raysCoarse, self.lxyz)
		dirCoarseShape = tf.shape(raysCoarse[..., :3])
		dirsCoarse = tf.broadcast_to(raysDirCoarse[..., None, :],
			shape=dirCoarseShape)
		dirsCoarse = self.encoderFn(dirsCoarse, self.lDir)
		# compute the predictions from the coarse model
		(rgbCoarse, sigmaCoarse) = self.coarseModel([raysCoarse,
			dirsCoarse])
		
		# render the image from the predictions
		renderCoarse = self.renderImageDepth(rgb=rgbCoarse,
			sigma=sigmaCoarse, tVals=tValsCoarse)
		(_, _, weightsCoarse) = renderCoarse
		# compute the middle values of t vals
		tValsCoarseMid = (0.5 * 
			(tValsCoarse[..., 1:] + tValsCoarse[..., :-1]))
		# apply hierarchical sampling and get the t vals for the fine
		# model
		tValsFine = self.samplePdf(tValsMid=tValsCoarseMid,
			weights=weightsCoarse, nF=self.nF)
		tValsFine = tf.sort(
			tf.concat([tValsCoarse, tValsFine], axis=-1), axis=-1)
		# build the fine rays and positional encode it
		raysFine = (raysOriCoarse[..., None, :] + 
			(raysDirCoarse[..., None, :] * tValsFine[..., None]))
		raysFine = self.encoderFn(raysFine, self.lxyz)
		
		# build the fine directions and positional encode it
		dirsFineShape = tf.shape(raysFine[..., :3])
		dirsFine = tf.broadcast_to(raysDirCoarse[..., None, :],
			shape=dirsFineShape)
		dirsFine = self.encoderFn(dirsFine, self.lDir)
		# compute the predictions from the fine model
		rgbFine, sigmaFine = self.fineModel([raysFine, dirsFine])
		
		# render the image from the predictions
		renderFine = self.renderImageDepth(rgb=rgbFine,
			sigma=sigmaFine, tVals=tValsFine)
		(imageFine, _, _) = renderFine
		# compute the photometric loss and psnr
		lossFine = self.lossFn(images, imageFine)
		psnr = tf.image.psnr(images, imageFine, max_val=1.0)
		# compute the loss and psnr metrics
		self.lossTracker.update_state(lossFine)
		self.psnrMetric.update_state(psnr)
		# return the loss and psnr metrics
		return {"loss": self.lossTracker.result(),
			"psnr": self.psnrMetric.result()}
	@property
	def metrics(self):
		# return the loss and psnr tracker
		return [self.lossTracker, self.psnrMetric]
	

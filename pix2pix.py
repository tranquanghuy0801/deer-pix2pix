from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
from losses import categorical_focal_loss
import numpy as np
import os
import wandb
import cv2
from utils import score_predictions
from models import resnet50_segnet

LABELMAP_DEER = {
	0 : (0, 0, 128),
	1 : (0, 0, 0),
	2 : (0, 128, 0),
	3 : (0,  128,  128),
}

LABELMAP_RGB = { k: (v[2], v[1], v[0]) for k, v in LABELMAP_DEER.items() }

class Pix2Pix():
	def __init__(self):
		# Input shape
		self.img_rows = 256
		self.img_cols = 256
		self.channels = 4
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		# Configure data loader
		self.dataset_name = 'semi-gan-final'
		self.data_loader = DataLoader(dataset_name=self.dataset_name,
									  img_res=(self.img_rows, self.img_cols))


		# Calculate output shape of D (PatchGAN)
		patch = int(self.img_rows / 2**4)
		self.disc_patch = (patch, patch, 1)

		# Number of filters in the first layer of G and D
		self.gf = 64
		self.df = 64
		self.n_critic = 5
		self.clip_value = 1
		# optimizer = Adam(0.0002, 0.5)
		
		# g_optimizer = RMSprop(lr=0.00005)
		g_optimizer = Adam(0.02, 0.5)
		# d_optimizer = RMSprop(lr=0.0005)
		d_optimizer = Adam(0.002, 0.5)
		alpha_fl = [0.30, 0.50, 0.15, 0.05]
		gamma_fl = 2

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss=['mse'],
			optimizer=d_optimizer,
			metrics=['accuracy'])

		#-------------------------
		# Construct Computational
		#   Graph of Generator
		#-------------------------

		# Build the generator
		# self.generator = self.build_generator()
		self.generator = resnet50_segnet(4,input_height=self.img_rows,input_width=self.img_cols)

		# Input images and their conditioning images
		img_A = Input(shape=self.img_shape)
		img_B = Input(shape=(self.img_rows, self.img_cols, 3))

		# By conditioning on B generate a fake version of A
		fake_A = self.generator(img_B)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# Discriminators determines validity of translated images / condition pairs
		valid = self.discriminator([fake_A, img_B])

		self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
		self.combined.compile(loss=['mse', categorical_focal_loss(alpha=alpha_fl,gamma=gamma_fl)],
							  loss_weights=[1, 10],
							  optimizer=g_optimizer)

	def build_generator(self):
		"""U-Net Generator"""

		def conv2d(layer_input, filters, f_size=4, bn=True):
			"""Layers used during downsampling"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
			"""Layers used during upsampling"""
			u = UpSampling2D(size=2)(layer_input)
			u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
			if dropout_rate:
				u = Dropout(dropout_rate)(u)
			u = BatchNormalization(momentum=0.8)(u)
			u = Concatenate()([u, skip_input])
			return u

		# Image input
		d0 = Input(shape=(self.img_rows, self.img_cols, 3))

		# Downsampling
		d1 = conv2d(d0, self.gf, bn=False)
		print(d1)
		d2 = conv2d(d1, self.gf*2)
		d3 = conv2d(d2, self.gf*4)
		d4 = conv2d(d3, self.gf*8)
		d5 = conv2d(d4, self.gf*8)
		d6 = conv2d(d5, self.gf*8)
		d7 = conv2d(d6, self.gf*8)

		# Upsampling
		u1 = deconv2d(d7, d6, self.gf*8)
		u2 = deconv2d(u1, d5, self.gf*8)
		u3 = deconv2d(u2, d4, self.gf*8)
		u4 = deconv2d(u3, d3, self.gf*4)
		u5 = deconv2d(u4, d2, self.gf*2)
		u6 = deconv2d(u5, d1, self.gf)

		u7 = UpSampling2D(size=2)(u6)
		output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
		return Model(d0, output_img)

	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, bn=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		img_A = Input(shape=self.img_shape)
		img_B = Input(shape=(self.img_rows, self.img_cols, 3))

		# Concatenate image and conditioning image by channels to produce input
		combined_imgs = Concatenate(axis=-1)([img_A, img_B])

		d1 = d_layer(combined_imgs, self.df, bn=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)

		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

		return Model([img_A, img_B], validity)
	
	def write_log(self, epoch, batch, G_loss, D_loss):
		f = open('images/{}/loss.txt'.format(self.dataset_name),"a")
		f.write("{}-{}-{}-{}-{}-{}-{}\n".format(epoch, batch, D_loss[0], D_loss[1] * 100, G_loss[0], G_loss[1], G_loss[2]))
		f.close()

	def train(self, epochs, batch_size=1, sample_interval=50):

		start_time = datetime.datetime.now()
		best_d_loss = 999
		best_g_loss = 999

		# Adversarial loss ground truths
		valid = np.ones((batch_size,) + self.disc_patch)
		fake = np.zeros((batch_size,) + self.disc_patch)

		for epoch in range(epochs):
			for batch_i, (imgs_A, imgs_B, _) in enumerate(self.data_loader.load_batch(batch_size, is_testing=False)):

				if batch_i % 5 == 0:
				# for _ in range(self.n_critic):

				# ---------------------
				#  Train Discriminator
				# ---------------------

					# Condition on B and generate a translated version
					fake_A = self.generator.predict(imgs_B)

					# Train the discriminators (original images = real / generated = Fake)
					d_loss_real = self.discriminator.train_on_batch([np.array(imgs_A), np.array(imgs_B)], np.array(valid))
					d_loss_fake = self.discriminator.train_on_batch([np.array(fake_A), np.array(imgs_B)], np.array(fake))
					d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
					# Clip critic weights
					# for l in self.discriminator.layers:
					# 	weights = l.get_weights()
					# 	weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
					# 	l.set_weights(weights)


				# -----------------
				#  Train Generator
				# -----------------

				# Train the generators
				g_loss = self.combined.train_on_batch([np.array(imgs_A), np.array(imgs_B)], [np.array(valid), np.array(imgs_A)])
				elapsed_time = datetime.datetime.now() - start_time
				self.write_log(epoch, batch_i, g_loss, d_loss)

				# if d_loss[0] < best_d_loss:
				# 	best_d_loss = d_loss[0]
				# 	# print("Saving best discriminator model at epoch %d/%d - [D loss: %f, acc: %3d%%]" % (epoch, epochs,  d_loss[0], 100*d_loss[1]))
				# 	self.discriminator.save_weights("saved_model/%s/%s" % (self.dataset_name, 'best-d-' + str(epoch) + '.h5'))
				# if g_loss[1] < best_g_loss:
				# 	best_g_loss = g_loss[-1]
				# 	# print("Saving best generator model at epoch %d/%d - [G loss: %f, acc: %3d%%]" % (epoch, epochs,  g_loss[0], 100*g_loss[1]))
				# 	self.generator.save_weights("saved_model/%s/%s" % (self.dataset_name, 'best-g-' + str(epoch) + '.h5'))  
				# Plot the progress
				print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
																		batch_i, self.data_loader.n_batches,
																		d_loss[0], 100*d_loss[1],
																		g_loss[-1],
																		elapsed_time))														

				# If at save interval => save generated image samples
				if batch_i % sample_interval == 0:
					self.sample_images(epoch, batch_i)
		# Save final weights
		# self.discriminator.save_weights("saved_model/%s/%s" % (self.dataset_name, 'best-d-final.h5'))
		# self.generator.save_weights("saved_model/%s/%s" % (self.dataset_name, 'best-g-final.h5'))  

	def sample_images(self, epoch, batch_i):
		os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

		if not os.path.exists('images/%s/output' % self.dataset_name ):
			os.mkdir('images/%s/output' % self.dataset_name )
		r, c = 3, 1

		_, imgs_B, imgs = self.data_loader.load_data(batch_size=1, is_testing=True)
		fake_A = self.generator.predict(imgs_B)
		print(fake_A.shape)
		fake_A_new = np.zeros((fake_A.shape[0],fake_A.shape[1],fake_A.shape[2],3))
		for i in range(c):
			for a in range(0, fake_A[i].shape[0]):
				for b in range(0, fake_A[i].shape[1]):
					fake_img = np.zeros(fake_A[i].shape)
					fake_A_new[i][a, b, :] = LABELMAP_RGB[np.argmax(fake_A[i][a,b,:])]


		# Rescale images 0 - 1
		# gen_imgs = ((gen_imgs * 0.5) + 0.5) * 255
		imgs_B = (imgs_B + 1) * 127.5

		img_A_new = np.zeros((1, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
		for j in range(c):
			for a in range(0, imgs[j].shape[0]):
				for b in range(0, imgs[j].shape[1]):
					img_A_new[j][a, b, :][0] = imgs[j][a, b, :][2]
					img_A_new[j][a, b, :][1] = imgs[j][a, b, :][1]
					img_A_new[j][a, b, :][2] = imgs[j][a, b, :][0]

		gen_imgs = np.concatenate([imgs_B, fake_A_new, img_A_new])
		titles = ['Condition', 'Generated', 'Original']
		fig, axs = plt.subplots(r, c)
		cnt = 0
		
		# Rescale images 0 - 1
		for i in range(r):
			for j in range(c):
				if i == 1:
					cv2.imwrite("images/%s/output/generated_%d_%d_%d.png" % (self.dataset_name, cnt, epoch, batch_i),gen_imgs[cnt].astype(np.uint8))
				if i == 2:
					cv2.imwrite("images/%s/output/original_%d_%d_%d.png" % (self.dataset_name, cnt, epoch, batch_i),gen_imgs[cnt].astype(np.uint8))
				axs[i].imshow(gen_imgs[cnt].astype(np.uint8))
				axs[i].set_title(titles[i])
				axs[i].axis('off')
				cnt += 1
				# if (i == 0):
				# 	axs[i,j].imshow(imgs_B[j].astype(np.uint8))
				# if (i == 1):
				# 	fake_A_new = np.zeros((fake_A[j].shape[0], fake_A[j].shape[1], 3))
				# 	for a in range(0, fake_A[j].shape[0]):
				# 		for b in range(0, fake_A[j].shape[1]):
				# 			fake_A_new[a, b, :] = LABELMAP_RGB[np.argmax(fake_A[j][a,b,:])]
				# 	cv2.imwrite("images/%s/output/generated_%d_%d_%d.png" % (self.dataset_name, cnt, epoch, batch_i),fake_A_new.astype(np.uint8))
				# 	axs[i,j].imshow(fake_A_new.astype(np.uint8))
				# if (i == 2):
				# 	img_A_new = np.zeros((imgs[j].shape[0], imgs[j].shape[1], 3))
				# 	for a in range(0, imgs[j].shape[0]):
				# 		for b in range(0, imgs[j].shape[1]):
				# 			img_A_new[a, b, :][0] = imgs[j][a, b, :][2]
				# 			img_A_new[a, b, :][1] = imgs[j][a, b, :][1]
				# 			img_A_new[a, b, :][2] = imgs[j][a, b, :][0]
				# 	cv2.imwrite("images/%s/output/original_%d_%d_%d.png" % (self.dataset_name, cnt, epoch, batch_i),img_A_new.astype(np.uint8))
				# 	axs[i,j].imshow(img_A_new.astype(np.uint8))
				# axs[i, j].set_title(titles[i])
				# axs[i,j].axis('off')
				# cnt += 1
		fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
		# fig.savefig(os.path.join(wandb.run.dir, "%d_%d.png" % (epoch, batch_i)))
		plt.close()


if __name__ == '__main__':
	gan = Pix2Pix()
	epochs=30
	batch_size=20
	sample_interval=15
	config = {
		'epochs' : epochs,
		'batch_size': batch_size,
		'sample_interval': sample_interval
	}
	# wandb.init(config=config)
	gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
	score, _ = score_predictions("semi-gan-final")
	print(score)

import cv2
from glob import glob
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# Color (BGR) to class
INV_LABELMAP_DEER = {
	(0,  0, 128) : 0,
	(0,  0, 0) : 1,
	(0, 128, 0) : 2,
	(0, 128, 128) : 3,
}

COLORS = [k for k, v in INV_LABELMAP_DEER.items()]

class DataLoader():
	def __init__(self, dataset_name, img_res=(128, 128)):
		self.dataset_name = dataset_name
		self.img_res = img_res

	def load_data(self, batch_size=1, is_testing=False):
		data_type = "train" if not is_testing else "test"
		if is_testing:
			path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
		else:
			path = glob('./datasets/%s/%s/images/*' % (self.dataset_name, data_type))

		batch_images = np.random.choice(path, size=batch_size)

		imgs_A = []
		imgs_B = []
		imgs = []
		for img_path in batch_images:
			img = self.imread(img_path)

			h, w, _ = img.shape
			_w = int(w/2)
			img_B, img_A = img[:, :_w, :], img[:, _w:, :]
			# img = self.imread(os.path.join(img_path))

			imgs_A.append(img_A)
			imgs_B.append(img_B)
			imgs.append(img_A)

		# imgs_A = np.array(imgs_A)/127.5 - 1.
		imgs_B = np.array(imgs_B)/127.5 - 1.

		return imgs_A, imgs_B, imgs

	def load_batch(self, batch_size=1, is_testing=False):
		data_type = "train" if not is_testing else "val"
		path = glob('./datasets/%s/%s/images/*' % (self.dataset_name, data_type))

		self.n_batches = int(len(path) / batch_size)

		for i in range(self.n_batches-1):
			batch = path[i*batch_size:(i+1)*batch_size]
			imgs_A, imgs_B, imgs = [], [], []
			for img in batch:
				# img = self.imread(img)
				# h, w, _ = img.shape
				# half_w = int(w/2)
				img_B = self.imread(img)
				img_B = cv2.resize(img_B, self.img_res)
				filename = img.split('/')
				label_filename = filename[len(filename) - 1]
				label_filename = 'extra-'.join(label_filename.split('extra'))
				label_path = os.path.join("datasets", self.dataset_name, data_type, "labels", label_filename)
				if os.path.exists(label_path):
					img_input = self.imread(label_path)
					img_input = np.array(cv2.resize(img_input, self.img_res))
					img_A = np.zeros((img_input.shape[0], img_input.shape[1], 4))
					for i in range(0, img_input.shape[0]):
						for j in range(0, img_input.shape[1]):
							a = np.zeros((1, 4))
							arr = np.array(COLORS) - tuple(img_input[i,j,:])
							index = np.argmin(np.sum(np.abs(arr), axis=1))
							a[0][index] = 1 
							img_A[i, j, :] = a.astype(np.uint8)
					imgs.append(self.imread(label_path))
				else:
					img_A = np.zeros((img_B.shape[0], img_B.shape[1], 4)) * (-1)
					imgs.append('')
				# img_A = to_categorical(img_input[:, :, 0], 4)

				if not is_testing and np.random.random() > 0.5:
						img_A = np.fliplr(img_A)
						img_B = np.fliplr(img_B)

				imgs_A.append(img_A)
				imgs_B.append(img_B)

			# imgs_A = np.array(imgs_A)/127.5 - 1.
			imgs_B = np.array(imgs_B)/127.5 - 1.

			yield imgs_A, imgs_B, imgs


	def imread(self, path):
		return cv2.imread(path).astype(np.float)

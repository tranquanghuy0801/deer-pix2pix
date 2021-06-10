import os
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob

LABELS_DEER = ['ground','tree','water','other']
LABELMAP_DEER = {
	0 : (0, 0, 128),
	1 : (0, 0, 0),
	2 : (0, 128, 0),
	3 : (0,  128,  128),
}

# Color (BGR) to class
INV_LABELMAP_DEER = {
	(0,  0, 128) : 0,
	(0,  0, 0) : 1,
	(0, 128, 0) : 2,
	(0, 128, 128) : 3,
}


def wherecolor(img, color, negate = False):

	k1 = (img[:, :, 0] == color[0])
	k2 = (img[:, :, 1] == color[1])
	k3 = (img[:, :, 2] == color[2])

	if negate:
		return np.where( not (k1 & k2 & k3) )
	else:
		return np.where( k1 & k2 & k3 )

def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=True,
						  title=None,
						  cmap=plt.cm.Blues,
						  savedir="predictions"):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)

	# Only use the labels that appear in the data
	labels_used = unique_labels(y_true, y_pred)
	classes = classes[labels_used]

	# Normalization with generate NaN where there are no ground label labels but there are predictions x/0
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)

	base, fname = os.path.split(title)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=classes, yticklabels=classes,
		   title=fname,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")

	plt.xlim([-0.5, cm.shape[1] - 0.5])
	plt.ylim([-0.5, cm.shape[0]- 0.5])

	fig.tight_layout()
	# save to directory
	if not os.path.isdir(savedir):
		os.mkdir(savedir)
	savefile = title
	# plt.savefig(savefile)
	return savefile, cm,labels_used

def score_masks(labelfile, predictionfile):

	label = cv2.imread(labelfile)
	label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
	prediction = cv2.imread(predictionfile)
	prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

	shape = label.shape[:2]

	label_class = np.zeros(shape, dtype='uint8')
	pred_class  = np.zeros(shape, dtype='uint8')

	for color, category in INV_LABELMAP_DEER.items():
		locs = wherecolor(label, color)
		label_class[locs] = category

	for color, category in INV_LABELMAP_DEER.items():
		locs = wherecolor(prediction, color)
		pred_class[locs] = category

	label_class = label_class.reshape((label_class.shape[0] * label_class.shape[1]))
	pred_class = pred_class.reshape((pred_class.shape[0] * pred_class.shape[1]))

	# Remove all predictions where there is a IGNORE (magenta pixel) in the groud label and then shift labels down 1 index
	# not_ignore_locs = np.where(label_class != 0)
	# label_class = label_class[not_ignore_locs] 
	# pred_class = pred_class[not_ignore_locs] -

	precision = precision_score(label_class, pred_class, average='weighted')
	recall = recall_score(label_class, pred_class, average='weighted')
	f1 = f1_score(label_class, pred_class, average='weighted')
	print(f'precision={precision} recall={recall} f1={f1}')

	savefile, cm, labels_used = plot_confusion_matrix(label_class, pred_class, np.array(LABELS_DEER), title=predictionfile.replace(".png","") + "-cf-matrix.png")

	print("CM")
	print(cm)

	return precision, recall, f1, savefile,cm, labels_used

def score_predictions(dataset):

	num_classes = 4
	count_cf = [0, 0, 0, 0]
	scores = []

	precision = []
	recall = []
	f1 = []

	cf_matrix = np.zeros([num_classes,num_classes])
	predictions = []
	confusions = []


	#for scene in train_ids + val_ids + test_ids:
	# for scene in test_ids:
	for predsfile in glob.glob("images/" + dataset + "/output/generated*.png"):
		img_name = predsfile.split('/')[-1]
		list_img_name = img_name.split('_')
		list_img_name[0] = 'original'
		list_img_name[1] = str(int(list_img_name[1]) + 1)
		labelfile = "images/" + dataset + "/output/" + "_".join(list_img_name)

		if not os.path.exists(labelfile):
			continue

		if not os.path.exists(predsfile):
			continue

		a, b, c, savefile, cm, labels = score_masks(labelfile, predsfile)
		cm = np.where(np.isnan(cm), 0, cm)
		for i in range(0,len(labels)):
			if np.sum(cm[i]) > 0:
				count_cf[labels[i]] = count_cf[labels[i]] + 1
			for j in range(0,len(labels)):
				ind_i = int(labels[i])
				ind_j = int(labels[j])
				if np.sum(cm[i]) > 0:
					cf_matrix[ind_i][ind_j] =  cf_matrix[ind_i][ind_j] + cm[i][j]
		precision.append(a)
		recall.append(b)
		f1.append(c)

		predictions.append(predsfile)
		confusions.append(savefile)
	print(count_cf)
	for i in range(0,len(count_cf)):
		if count_cf[i] != 0:
			cf_matrix[i] = cf_matrix[i]/count_cf[i]
	classes = ['ground','tree','water','other']
	title = 'Confusion-Matrix'
	fig, ax = plt.subplots()
	im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)

	ax.set(xticks=np.arange(cf_matrix.shape[1]),
		   yticks=np.arange(cf_matrix.shape[0]),
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	normalize = True

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cf_matrix.max() / 2.
	for i in range(cf_matrix.shape[0]):
		for j in range(cf_matrix.shape[1]):
			ax.text(j, i, format(cf_matrix[i, j], fmt),
					ha="center", va="center",
					color="white" if cf_matrix[i, j] > thresh else "black")

	plt.xlim([-0.5, cf_matrix.shape[1] - 0.5])
	plt.ylim([-0.5, cf_matrix.shape[0]- 0.5])

	fig.tight_layout()
	# save to directory
	plt.savefig(os.path.join("images",dataset,title + '.png'))

	# Compute test set scores
	scores = {
		'f1_mean' : np.mean(f1),
		'f1_std'  : np.std(f1),
		'pr_mean' : np.mean(precision),
		'pr_std'  : np.std(precision),
		're_mean' : np.mean(recall),
		're_std'  : np.std(recall),
	}

	return scores, zip(predictions, confusions)

if __name__ == '__main__':
	score, _ = score_predictions('gan-data-new1')
	print(score)

from data_loader import COLORS
import os
import cv2
import numpy as np

dataset_dir = './datasets/semi-gan/train/images'
count_labels = [0, 0, 0, 0]

for img_path in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_path)
    print('processing', img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            arr = np.array(COLORS) - tuple(img[i,j,:])
            index = np.argmin(np.sum(np.abs(arr), axis=1))
            count_labels[index] += 1

print('Labels: Ground, Tree, Water, Other', count_labels)
from .constants import *
from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
import cv2
import os
import numpy as np
from CNNUtil import paths

class ImageGenerator(Sequence):

    def __init__(self, data_dir= 'D:\Data\iris_pattern\Original2', augmentations=None):
        self.total_paths, self.total_labels = self.get_total_data_path(data_dir)
        self.batch_size = FLG.BATCH_SIZE
        self.indices = np.random.permutation(len(self.total_paths))
        self.augment = augmentations

    def get_total_data_path(self, data_dir):
        total_paths, total_labels = [], []  # 이미지 path와 정답(label) 세트를 저장할 list

        image_paths = sorted(list(paths.list_images(data_dir)))
        for image_path in image_paths:
            total_paths.append(image_path)
            label = image_path.split(os.path.sep)[-2]
            total_labels.append(label)

        return total_paths, total_labels

    def findRegion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = cv2.boundingRect(contours[0])
        len = w if w > h else h
        dst = img[y: y + len, x: x + len]
        return dst

    def img_padding_2(self, img, LENGTH=FLG.WIDTH):
        blank_image = np.zeros((LENGTH, LENGTH, 3), np.uint8)
        (w, h) = (img.shape[0], img.shape[1])
        len = w if w > h else h
        if len > LENGTH:
            big_img = np.zeros((len, len, 3), np.uint8)
            big_img[0:  w, 0:  h] = img
            dst = cv2.resize(big_img, (LENGTH, LENGTH))
            blank_image = dst
        else:
            blank_image[0:  w, 0:  h] = img
        return blank_image

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = self.findRegion(image)
        # image = self.img_padding_2(image)
        image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
        if self.augment is not None:
            image = self.augment(image=image)['image']

        image = img_to_array(image)
        return image

    def encode_one_hot(self, y_batchs):
        y_batch_one_hots = []
        for y_batch in y_batchs:
            if y_batch =='defect':
                one_hot = [1, 0]
            elif y_batch =='lacuna':
                one_hot = [1, 0]
            elif y_batch =='normal':
                one_hot = [0, 1]
            elif y_batch =='spoke':
                one_hot = [1, 0]
            elif y_batch =='spot':
                one_hot = [1, 0]

            y_batch_one_hots.append(one_hot)

        y_batch_one_hots = np.array(y_batch_one_hots)

        return y_batch_one_hots

    def __len__(self):
        return len(self.total_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        x_batch, y_batch = [], []

        selected_paths = [self.total_paths[i] for i in batch_idx]
        y_batch = [self.total_labels[i] for i in batch_idx]

        for img_path in selected_paths:
            x_batch.append(self.load_image(img_path))

        x_batch = np.array(x_batch, dtype="float") / 255.0
        y_batch = np.array(y_batch)

        y_batch_one_hot = self.encode_one_hot(y_batch)

        # y_batch_one_hot = LabelBinarizer().fit_transform(y_batch)

        return x_batch, y_batch_one_hot

    def on_epoch_end(self):
        self.indices = np.random.permutation(len(self.total_paths))



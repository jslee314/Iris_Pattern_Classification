from .constants import *
from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
import cv2
import os
import numpy as np
from CNNUtil import paths

class ImageGenerator(Sequence):

    def __init__(self, data_dir= 'D:\Data\iris_pattern\Original2'):
        self.total_paths, self.total_labels = self.get_total_data_path(data_dir)
        self.batch_size = FLG.BATCH_SIZE
        self.indices = np.random.permutation(len(self.total_paths))

    def get_total_data_path(self, data_dir):
        total_paths, total_labels = [], []  # 이미지 path와 정답(label) 세트를 저장할 list

        image_paths = sorted(list(paths.list_images(data_dir)))
        for image_path in image_paths:
            # a. 이미지 전체 파일 path 저장
            total_paths.append(image_path)
            # b. 이미지 파일 path에서  이미지의 정답(label) 세트 추출
            label = image_path.split(os.path.sep)[-2]
            total_labels.append(label)

        print('total_paths : ' + str(len(total_paths)))
        print('total_labels : ' + str(len(total_labels)))

        return total_paths, total_labels

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
        image = img_to_array(image)
        return image

    def __len__(self):
        return len(self.total_paths) // self.batch_size

    def __getitem__(self, idx):
        print('index : ' + str(idx))
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        print('batch_idx len : ' + str(len(batch_idx)))
        x_batch, y_batch = [], []

        selected_paths = [self.total_paths[i] for i in batch_idx]
        y_batch = [self.total_labels[i] for i in batch_idx]

        # print('total_paths : ' + str(selected_paths))
        # print('y batch : ' + str(y_batch))

        for img_path in selected_paths:
            x_batch.append(self.load_image(img_path))

        x_batch = np.array(x_batch, dtype="float") / 255.0
        y_batch = np.array(y_batch)

        lb = LabelBinarizer()
        y_batch = lb.fit_transform(y_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.random.permutation(len(self.img_paths))


class ImageGenerator(Sequence):

        def __init__(self, data_dir='D:\Data\iris_pattern\Original2'):
            self.total_paths, self.total_labels = self.get_total_data_path(data_dir)
            self.batch_size = FLG.BATCH_SIZE
            self.indices = np.random.permutation(len(self.total_paths))

        def get_total_data_path(self, data_dir):
            total_paths, total_labels = [], []  # 이미지 path와 정답(label) 세트를 저장할 list

            image_paths = sorted(list(paths.list_images(data_dir)))
            for image_path in image_paths:
                # a. 이미지 전체 파일 path 저장
                total_paths.append(image_path)
                # b. 이미지 파일 path에서  이미지의 정답(label) 세트 추출
                label = image_path.split(os.path.sep)[-2]
                total_labels.append(label)

            total_labels = np.array(total_labels)
            lb = LabelBinarizer()
            total_labels = lb.fit_transform(total_labels)

            return total_paths, total_labels

        def load_image(self, image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
            image = img_to_array(image)
            return image

        def __len__(self):
            return len(self.total_paths) // self.batch_size

        def __getitem__(self, idx):
            print('index : ' + str(idx))
            batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            print('batch_idx len : ' + str(len(batch_idx)))
            x_batch, y_batch = [], []

            selected_paths = [self.total_paths[i] for i in batch_idx]
            y_batch = [self.total_labels[i] for i in batch_idx]

            # print('total_paths : ' + str(selected_paths))
            # print('y batch : ' + str(y_batch))

            for img_path in selected_paths:
                x_batch.append(self.load_image(img_path))

            x_batch = np.array(x_batch, dtype="float") / 255.0
            y_batch = np.array(y_batch)

            lb = LabelBinarizer()
            y_batch = lb.fit_transform(y_batch)
            # print('y batch : ' + str(y_batch))
            return x_batch, y_batch

        def on_epoch_end(self):
            self.indices = np.random.permutation(len(self.img_paths))
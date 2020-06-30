import random
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from CNNUtil import paths
from .constants import *

import numpy as np

class DataLoader:

    kernel_sharpen = np.array(
        [[-1, -1, -1, -1, -1],
         [-1, 2, 2, 2, -1],
         [-1, 2, 8, 2, -1],
         [-1, 2, 2, 2, -1],
         [-1, -1, -1, -1, -1]]) / 8.0  # 정규화위해 8로나눔

    def img_preprocess(img):
        # cv2.imshow('orig', img)
        # cv2.waitKey(1000)

        # 3. 샤프닝
        #img = cv2.filter2D(img, -1, DataLoader.kernel_sharpen)

        # 1. gray
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 히스토그램 균일화(대비조정)
        #img = cv2.equalizeHist(img)

        # 4. 리사이즈
        new_img = cv2.resize(img, (FLG.HEIGHT, FLG.WIDTH))

        # cv2.imshow('before', img)
        # cv2.waitKey(1000)
        #
        # cv2.imshow('after', new_img)
        # cv2.waitKey(1500)
        return new_img

    @staticmethod
    def load_data(dir=dir):
        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images(dir)))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        datas = []                  # 이미지 파일을 array 형태로 변환해서 저장할 list
        labels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            #print(imagePath)
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = DataLoader.img_preprocess(image)
            image = img_to_array(image)
            datas.append(image)
            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)





        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        data = np.array(datas, dtype="float") / 255.0
        labels = np.array(labels)

        print("[INFO] data matrix: {} images ({:.2f}MB)".format( len(imagePaths), data.nbytes / (1024 * 1000.0)))

        #  훈련 용 데이터의 80 %와 시험용 나머지 20 %를 사용하여 데이터를 훈련 및 테스트 분할로 분할
        (x_train, x_val, y_train, y_val) = train_test_split(data, labels,
                                                            test_size=0.2, random_state=42)

        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.transform(y_val)  # fit_transform??????????

        return x_train, y_train, x_val, y_val, lb

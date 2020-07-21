import random
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from CNNUtil import paths
from .constants import *
import numpy as np
from .gendataloader import ImageGenerator

class DataLoader:

    # kernel_sharpen = np.array(
    #     [[-1, -1, -1, -1, -1],
    #      [-1, 2, 2, 2, -1],
    #      [-1, 2, 8, 2, -1],
    #      [-1, 2, 2, 2, -1],
    #      [-1, -1, -1, -1, -1]]) / 8.0  # 정규화위해 8로나눔
    # def findRegion(img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #     contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     len = w if w > h else h
    #     dst = img[y: y + len, x: x + len]
    #     return dst
    #
    # def img_padding_2(img, LENGTH=FLG.WIDTH):
    #     blank_image = np.zeros((LENGTH, LENGTH, 3), np.uint8)
    #     (w, h) = (img.shape[0], img.shape[1])
    #     len = w if w > h else h
    #     if len > LENGTH:
    #         big_img = np.zeros((len, len, 3), np.uint8)
    #         big_img[0:  w, 0:  h] = img
    #         dst = cv2.resize(big_img, (LENGTH, LENGTH))
    #         blank_image = dst
    #     else:
    #         blank_image[0:  w, 0:  h] = img
    #     return blank_image

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
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = DataLoader.findRegion(image)
            # image = DataLoader.img_padding_2(image)
            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
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
        (x_train, x_val, y_train, y_val) = train_test_split(data, labels, test_size=0.2, random_state=42)

        y_train = ImageGenerator.encode_one_hot(1, y_train)
        y_val = ImageGenerator.encode_one_hot(1, y_val)


        return x_train, y_train, x_val, y_val


    @staticmethod
    def test_load_data(dir=dir):
        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images(dir)))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        datas = []                  # 이미지 파일을 array 형태로 변환해서 저장할 list
        labels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
            image = img_to_array(image)
            datas.append(image)
            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        x_val = np.array(datas, dtype="float") / 255.0
        y_val = np.array(labels)
        y_val = ImageGenerator.encode_one_hot(1, y_val)

        return x_val, y_val
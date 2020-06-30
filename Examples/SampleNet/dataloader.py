from imutils import paths
import numpy as np
import random
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class DataLoader:

    def __init__(self, data_path):
        X_datas = []
        Y_datas = []

        # 이미지 경로를 잡고 무작위로 섞음
        X_datas_paths = sorted(list(paths.list_images(data_path)))
        random.seed(42)
        random.shuffle(X_datas_paths)

        # 해당 path에 있는 입력 이미지들을 loop over 하면서 X_ data , Y_data list에 append 함
        for X_datas_path in X_datas_paths:
            '''이미지를 32x32로 조정(비율무시)하고 32x32x3픽셀이미지로 평평하게'''
            X_data = cv2.imread(X_datas_path)
            X_data = cv2.resize(X_data, (32, 32)).flatten()
            X_datas.append(X_data)

            '''해당 path의 폴더를 보고 Y_data 즉 해당 클래스의 라벨 label을 추출'''
            Y_data = X_datas_path.split(os.path.sep)[-2]
            Y_datas.append(Y_data)

        # 0-1사이로 normalization 시킨 후 list를  ndarray 형식으로 변환
        self.X_datas = np.array(X_datas, dtype="float") / 255.0
        self.Y_datas = np.array(Y_datas)
        '''  self.X_datas : [[0.18235435 0.25235435 ...] [0.24235435 0.5235435...] ... ]
            self.Y_datas : ['panda' 'dogs' 'cats' 'dogs' .....] 
        '''

    def train_load(self, test_ratio, random_state):
        # train 75%, val 25%
        (x_train, x_val, y_train, y_val) = train_test_split(self.X_datas, self.Y_datas, test_size=test_ratio, random_state=random_state)

        #  라벨링 전환
        '''  ['panda' 'dogs' 'cats' 'dogs' .....]  -> [ [0 1 0]\n [1 0 0 ]\n [0 0 1 ]\n .... ]  '''
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.transform(y_val)

        '''  lb.classes_ : ['cats' 'cogs' 'panda']  ndarray 형식  '''

        return x_train, y_train, x_val, y_val, lb

    def test_load(self):
        ''' 추후 구현 예정~!!  '''
        x_test = []
        y_test = []

        return x_test, y_test

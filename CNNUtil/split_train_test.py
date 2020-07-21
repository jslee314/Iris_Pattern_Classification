import os
import shutil
from CNNUtil import paths

import random

label_dic = {'defect': [],
             'lacuna': [],
             'normal': [],
             'spoke': [],
             'spot': [],
             'unknown': []}

# data_path = 'C:/Users/jslee/Downloads/1st_pattern_dataset_for_50/mask'
data_path = 'D:/2. data/js_15angle/mask'
dst_path = 'D:/2. data/iris_pattern/Region_15/js_0721_1400_5'


for (root, dirs, files) in os.walk(data_path):
    for file in files:
        label = os.path.dirname(os.path.join(root, file)).split('\\')[-1]
        for key in label_dic:
            if label == key:
                label_dic[key].append(os.path.join(root, file))




for key in label_dic:
    labels = label_dic[key]
    random.seed(42)
    random.shuffle(labels)
    train_path, test_path = labels[:int(len(labels) * 0.8)], labels[int(len(labels) * 0.8):]

    for idx, path in enumerate(train_path):
        train_new_path = os.path.join(dst_path, 'train', os.path.dirname(path).split('\\')[-1])
        if not os.path.exists(train_new_path):
            os.makedirs(train_new_path)
        shutil.copy(path, train_new_path + '/')

    for idx, path in enumerate(test_path):
        test_new_path = os.path.join(dst_path, 'test', os.path.dirname(path).split('\\')[-1])
        if not os.path.exists(test_new_path):
            os.makedirs(test_new_path)
        shutil.copy(path, test_new_path + '/')

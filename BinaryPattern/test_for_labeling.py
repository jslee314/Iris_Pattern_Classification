from BinaryPattern.util.constants import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from keras import backend as K
import shutil
import os
from CNNUtil import paths
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet
import copy

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT,
                            depth=FLG.DEPTH, classes=FLG.CLASS_NUM, finalAct="softmax")

h5_weights_path = 'output/smallervgg_0721_1206_5_200/model_saved/smallervgg_0721_1206_5_200_32_weight.h5'

model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

datas = []
origs = []
src_dir = 'D:/2. data/js_15angle/mask_blue/'
dst_dir = 'D:/2. data/js_15angle/predicted_mask/'

imagePaths = sorted(list(paths.list_images(src_dir)))
for imagePath in imagePaths:
    print(imagePath)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
    origs.append(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    datas.append(image)

data = np.array(datas, dtype="float") / 255.0
preds = model.predict(data, batch_size=FLG.BATCH_SIZE)

label_lists = ['defect', 'lacuna', 'normal', 'spoke', 'spot']

for i, prediction in enumerate(preds):
    classIdx = np.argmax(preds[i])
    img_name = imagePath.split(os.path.sep)[-1]

    shutil.copy(src_dir+img_name, dst_dir+label_lists[classIdx]+'/'+img_name)

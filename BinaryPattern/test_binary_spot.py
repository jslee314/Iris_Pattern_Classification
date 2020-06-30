from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from CNNUtil.imutils import imutils
import cv2
from tensorflow.keras.losses import categorical_crossentropy
from BinaryPattern.util.constants import *
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet

import random
from CNNUtil import paths

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst
def img_padding_2(img, LENGTH=FLG.WIDTH):
    blank_image = np.zeros((LENGTH, LENGTH, 3), np.uint8)
    (w, h)=(img.shape[0], img.shape[1])
    len = w if w > h else h
    if len>LENGTH:
        big_img = np.zeros((len, len, 3), np.uint8)
        big_img[0:  w,  0:  h] = img
        dst = cv2.resize(big_img, (LENGTH, LENGTH))
        blank_image = dst
    else:
        blank_image[0:  w, 0:  h] = img
    return blank_image
model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT,depth= FLG.DEPTH, classes=2, finalAct="softmax")

# h5_weights_path = './output/a_spot300_16/modelsaved/h5/a_spot300_16_weights.h5'
h5_weights_path = './output/a_spot_padding150_16/modelsaved/h5/a_spot_padding150_16_weights.h5'

# h5_weights_path = './util/binary_smallVGG_spot_x4_400_16.h5'

model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
datas = []
origs = []
data_dir = 'D:/Data/iris_pattern/test_image/11'
imagePaths = sorted(list(paths.list_images(data_dir)))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = findRegion(image)
    image = img_padding_2(image)
    origs.append(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    datas.append(image)
data = np.array(datas, dtype="float") / 255.0
predictions = model.predict(data, batch_size=FLG.BATCH_SIZE)
for i, prediction in enumerate(predictions):
    (spot, normal) = (prediction[0], prediction[1])
    labels = []
    labels.append("{}: {:.2f}%".format('spot', spot * 100))
    labels.append("{}: {:.2f}%".format('normal', normal * 100))
    output = imutils.resize(origs[i], width=400)
    cv2.putText(output, labels[1], (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, labels[0], (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if normal > 0.5:
        cv2.putText(output, labels[1], (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, labels[0], (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Output", output)
    cv2.waitKey(0)




    # import cv2
    # from CNNUtil import paths
    # def findRegion(img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #     contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     len = w if w > h else h
    #     dst = img[y: y + len, x: x + len]
    #     return dst
    # data_dir = 'D:/Data/iris_pattern/test_image/11'
    # imagePaths = sorted(list(paths.list_images(data_dir)))
    # for imagePath in imagePaths:
    #     image = cv2.imread(imagePath)
    #     image = findRegion(image)
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     print(imagePath)
    #     cv2.imwrite(imagePath, image)
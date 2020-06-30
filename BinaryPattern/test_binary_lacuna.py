from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from CNNUtil.imutils import imutils
import cv2
from tensorflow.keras.losses import categorical_crossentropy
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet

from BinaryPattern.util.constants import *
import random
from CNNUtil import paths
from keras import backend as K

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst
input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)
model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=2)
# model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT,depth= FLG.DEPTH, classes=2, finalAct="softmax")

# h5_weights_path = './output/a_lacuna_x4_150_32/modelsaved/h5/a_lacuna_x4_150_32_weights.h5'
h5_weights_path = './util/tflite/mobileNetV2_lacuna_padding200_32_weights.h5'
model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
datas = []
origs = []
data_dir = 'D:/Data/iris_pattern/test_image/11'
# data_dir = 'D:/Data/iris_pattern/test_image/11'
imagePaths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(imagePaths)
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = findRegion(image)
    origs.append(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
    image = img_to_array(image)
    datas.append(image)
data = np.array(datas, dtype="float") / 255.0
predictions = model.predict(data, batch_size=FLG.BATCH_SIZE)

for i, prediction in enumerate(predictions):
    (lacuna, normal) = (prediction[0], prediction[1])
    labels = []
    labels.append("{}: {:.2f}%".format('lacuna', lacuna * 100))
    labels.append("{}: {:.2f}%".format('normal', normal * 100))
    print(labels[0])
    print(labels[1])
    # output = imutils.resize(origs[i], width=400)
    output = origs[i]
    # y = 300
    y = 300
    cv2.putText(output, labels[1], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, labels[0], (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if normal > 0.5:
        cv2.putText(output, labels[1], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, labels[0], (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
    else:
        cv2.imshow("Output", output)
        cv2.waitKey(0)


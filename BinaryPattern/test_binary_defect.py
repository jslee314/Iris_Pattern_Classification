from BinaryPattern.util.constants import *

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from keras import backend as K
from CNNUtil.imutils import imutils
from CNNUtil.gradcam import GradCAM
from CNNUtil import paths

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img

from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from CNNModels.VGG.model.vgg16v1 import VGG_16
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder
def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y:y+len, x:x+len]
    return dst
def img_padding_2(img, LENGTH=FLG.HEIGHT):
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

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

# model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=2)
model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=2, finalAct="softmax")

h5_weights_path = 'output/smallvgg_defect_padding200_32/modelsaved/h5/smallvgg_defect_padding200_32_weights.h5'
# h5_weights_path = 'output/a_defect_x4_300_32/modelsaved/h5/a_defect_x4_200_32_weights.h5'

model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

datas = []
origs = []

data_dir = 'D:/2. data/iris_pattern/test_image/22'
imagePaths = sorted(list(paths.list_images(data_dir)))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # image = findRegion(image)
    image = img_padding_2(image)
    origs.append(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    datas.append(image)

data = np.array(datas, dtype="float") / 255.0
predictions = model.predict(data, batch_size=FLG.BATCH_SIZE)

# 그라디언트 클래스 활성화 맵을 초기화하고 히트 맵을 빌드

for i, prediction in enumerate(predictions):
    (defect, normal) = (prediction[0], prediction[1])
    labels = []
    labels.append("{}: {:.2f}%".format('defect', defect * 100))
    labels.append("{}: {:.2f}%".format('normal', normal * 100))
    # output = imutils.resize(origs[i], width=400)
    output = origs[i]
    if defect > 0.5:
        cv2.putText(output, labels[1], (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, labels[0], (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)
    else:
        cv2.putText(output, labels[1], (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, labels[0], (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)


    print("[info] gradCam...")
    pred = np.argmax(prediction)
    cam = GradCAM(model, pred)

    image = np.expand_dims(datas[i], axis=0)
    image = imagenet_utils.preprocess_input(image)
    heatmap = cam.compute_heatmap(datas[i].reshape(1, 180, 180, 3))

    # 결과 히트 맵의 크기를 원래 입력 이미지 크기로 조정 한 다음 이미지 위에 히트 맵을 오버레이
    heatmap = cv2.resize(heatmap, (origs[i].shape[1], origs[i].shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, origs[i], alpha=0.5)

    # 출력 이미지에 예측 된 레이블을 그립니다.
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, labels[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 원본 이미지와 결과로 생성 된 히트 맵 및 출력 이미지를 화면에 표시
    output = np.vstack([origs[i], heatmap, output])
    output = imutils.resize(output, height=700)
    cv2.imshow("Output", output)
    cv2.waitKey(0)






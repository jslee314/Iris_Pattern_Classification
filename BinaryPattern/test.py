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

model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=2, finalAct="softmax")
h5_weights_path = 'output/smallvgg_defect_padding200_32/modelsaved/h5/smallvgg_defect_padding200_32_weights.h5'
model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

data_dir = 'D:/2. data/iris_pattern/test_image/22/20191016_e2c0b342-6099-42c0-8490-366b932a00se35.png'

# 디스크에서 원본 이미지를로드 한 다음 (OpenCV 형식) 이미지를 대상 크기로 조정
orig = cv2.imread(data_dir)
resized = cv2.resize(orig, (180, 180))

# 디스크에서 입력 이미지를로드하고 (Keras / TensorFlow 형식으로) 전처리
image = load_img(data_dir, target_size=(180, 180))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

# 네트워크를 사용하여 입력 image에 대해 예측하고 해당 확률이 가장 큰 클래스 레이블 예측
preds = model.predict(image)

# 사람이 읽을 수있는 레이블을 얻기 위해 ImageNet 예측을 디코딩
(prob, normal) = (preds[0][0], preds[0][1])
labels = []
label = "{}: {:.2f}%".format('defect', prob * 100)

# decoded = imagenet_utils.decode_predictions(preds)
# (imagenetID, label, prob) = decoded[0][0]
# label = "{}: {:.2f}%".format(label, prob * 100)
# print("[INFO] {}".format(label))


i = np.argmax(preds[0])
# 그라디언트 클래스 활성화 맵을 초기화하고 히트 맵을 빌드
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)

# 결과 히트 맵의 크기를 원래 입력 이미지 크기로 조정 한 다음 이미지 위에 히트 맵을 오버레이
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# 출력 이미지에 예측 된 레이블을 그립니다.
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 원본 이미지와 결과로 생성 된 히트 맵 및 출력 이미지를 화면에 표시
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)



from BinaryPattern.util.constants import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import img_to_array
from keras import backend as K
from CNNUtil.imutils import imutils
from CNNUtil.gradcam import GradCAM
from CNNUtil import paths
import numpy as np
import cv2

from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from CNNModels.VGG.model.vgg16v1 import VGG_16
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder
import copy
def findRegion(img):
    ori = copy.copy(img)
    width, height = (img.shape[0],img.shape[1])
    for h in range(height):
        for w in range(width):
            b, g, r = img[w][h]
            if (b == 255 and g == 0 and r == 0):
                img[w][h] = 0, 0, 0
            else:
                img[w][h] = 255, 255, 255
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # len = w if w > h else h
    # dst = img[y: y + len, x: x + len]
    dst = ori[y: y + h, x: x + w]
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
# input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
# if K.image_data_format() == "channels_first":
#     input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)
h5_weights_path = 'output/smallervgg_0721_1500_5_200/model_saved/smallervgg_0721_1500_5_200_weight.h5'
data_dir = 'D:/2. data/iris_pattern/Region_15/js_0721_1500_5/test/defect'



model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=FLG.CLASS_NUM, finalAct="softmax")
model.load_weights(h5_weights_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

datas, origs = [], []
imagePaths = sorted(list(paths.list_images(data_dir)))
for imagePath in imagePaths:
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
    if classIdx != 0:
        labels_dic, preds_dic = {}, {}
        print('----')
        for idx, label_list in enumerate(label_lists):
            # preds_dic[label_list] = preds[i][idx]
            # labels_dic[label_list] = "{}: {:.2f}%".format('defect', preds_dic(label_list) * 100)
            print("label / i: " + "{}: {:.2f}%".format(label_list, preds[i][idx] * 100))

        cam = GradCAM(model, classIdx)
        input_data = data[i].reshape([1, 180, 180, 3])
        heatmap = cam.compute_heatmap(input_data)

        # 결과 히트 맵의 크기를 원래 입력 이미지 크기로 조정 한 다음 이미지 위에 히트 맵을 오버레이
        heatmap = cv2.resize(heatmap, (origs[i].shape[1], origs[i].shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, origs[i], alpha=0.5)

        # 출력 이미지에 예측 된 레이블을 그립니다.
        cv2.putText(output, label_lists[classIdx], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 원본 이미지와 결과로 생성 된 히트 맵 및 출력 이미지를 화면에 표시
        output = np.vstack([origs[i], heatmap, output])
        output = imutils.resize(output, height=700)
        cv2.imshow("Output", output)
        cv2.waitKey(0)



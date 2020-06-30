from keras.preprocessing.image import img_to_array
import numpy as np
from CNNUtil.imutils import imutils
import cv2
from IrisPattern.util.dataloader import DataLoader
from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from IrisPattern.util.constants import *
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


print('[STEP 2] 모델 불러오기')
model, model_size = efficientNet_factory('efficientnet-b1',  load_weights=None, input_shape=(FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH), classes=5)
h5_weights_path = './output/data_load__Test_200_32/modelsaved/h5/data_load__Test_200_32_weights.h5'

model.load_weights(h5_weights_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print('[STEP 1] 이미지 불러오기')
datas = []
origs = []

data_dir = 'D:/Data/iris_pattern/test_image'
imagePaths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = findRegion(image)
    image = DataLoader.img_preprocess(image)
    origs.append(image.copy())
    image = img_to_array(image)
    datas.append(image)

data = np.array(datas, dtype="float") / 255.0



print('[STEP 3] 모델로 데이터 예측하기')
predictions = model.predict(data, batch_size=FLG.BATCH_SIZE)
# (defect_predictions, lacuna_predictions, normal_predictions, spoke_predictions, spot_predictions) =predictions
# print(str(len(defect_predictions)))
print('[STEP 4] 결과 보여주기')

# for i, (defect, lacuna, normal, spoke, spot) in enumerate(zip(defect_predictions, lacuna_predictions, normal_predictions, spoke_predictions, spot_predictions)):
for i, prediction in enumerate(predictions):
    (defect, lacuna, normal, spoke, spot) = (prediction[0], prediction[1], prediction[2], prediction[3], prediction[4])

    labels = []
    labels.append("{}: {:.2f}%".format('defect', defect * 100))
    labels.append("{}: {:.2f}%".format('lacuna', lacuna * 100))
    labels.append("{}: {:.2f}%".format('normal', normal * 100))
    labels.append("{}: {:.2f}%".format('spoke', spoke * 100))
    labels.append("{}: {:.2f}%".format('spot', spot * 100))

    # draw the label on the image
    green = (0, 255, 0)
    output = imutils.resize(origs[i], width=400)
    y = 300
    for label in labels:
        y = y + 20
        cv2.putText(output, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)

    cv2.imshow("Output", output)
    cv2.waitKey(0)


from CNNUtil.gradcam import GradCAM
from CNNUtil.imutils import imutils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import cv2

# initialize the model to be VGG16
Model = VGG16

# load the pre-trained CNN from disk
print("[INFO] loading model...")
model = Model(weights="imagenet")






# 디스크에서 원본 이미지를로드 한 다음 (OpenCV 형식) 이미지를 대상 크기로 조정
orig = cv2.imread("dog.jpg")
resized = cv2.resize(orig, (224, 224))

# 디스크에서 입력 이미지를로드하고 (Keras / TensorFlow 형식으로) 전처리
image = load_img("dog.jpg", target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

# 네트워크를 사용하여 입력 image에 대해 예측하고 해당 확률이 가장 큰 클래스 레이블 예측
preds = model.predict(image)
i = np.argmax(preds[0])

# 사람이 읽을 수있는 레이블을 얻기 위해 ImageNet 예측을 디코딩
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))



# 그라디언트 클래스 활성화 맵을 초기화하고 히트 맵을 빌드
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)

# 결과 히트 맵의 크기를 원래 입력 이미지 크기로 조정 한 다음 이미지 위에 히트 맵을 오버레이
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# 출력 이미지에 예측 된 레이블을 그립니다.
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)

# 원본 이미지와 결과로 생성 된 히트 맵 및 출력 이미지를 화면에 표시
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# 모델, 클래스 활성화 맵을 측정하는 데 사용되는 클래스 인덱스 및 클래스 활성화 맵을 시각화 할 때 사용할 레이어를 저장
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

		# 레이어 이름이 None이면 자동으로 대상 출력 레이어를 찾음
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# 네트워크 레이어를 역순으로 반복하여 네트워크에서 최종 컨볼 루션 레이어를 찾음
		for layer in reversed(self.model.layers):
			# layer에 4D 출력이 있는지 확인
			if len(layer.output_shape) == 4:
				return layer.name
		# 그렇지 않으면 4D 레이어를 찾을 수 없으므로 GradCAM 알고리즘을 적용 할 수 없음
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, image, eps=1e-8):
		# 다음을 적용하여 그라디언트 모델을 구성
		# (1) 사전 훈련 된 모델에 대한 입력, (2) 네트워크에서 (아마도) 최종 4D 레이어의 출력, (3) 모델에서 softmax 활성화의 출력
		gradModel = Model(inputs=[self.model.inputs],
						  outputs=[self.model.get_layer(self.layerName).output, self.model.output])

		# 자동 차별화를위한 기록 작업
		with tf.GradientTape() as tape:
			# 이미지 텐서를 float-32 데이터 유형으로 캐스팅하고, 그라디언트 모델을 통해 이미지를 전달하고, 특정 클래스 인덱스와 관련된 손실을 잡음
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]

		# 자동 미분을 사용하여 그라디언트 계산
		grads = tape.gradient(loss, convOutputs)

		# 유도 그라디언트 계산
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# 컨벌루션과 유도 그라디언트는 배치 차원 (필요하지 않은)을 가지므로 볼륨 자체를 잡고 배치를 폐기함
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# 기울기 값의 평균을 계산하고이를 가중치로 사용하여 가중치에 대한 필터의 숙고를 계산함
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# 입력 이미지의 공간 크기를 잡고 입력 이미지 크기와 일치하도록 출력 클래스 활성화 맵의 크기를 조정
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		# 모든 값이 [0, 1] 범위에 있도록 히트 맵을 정규화하고 결과 값을 [0, 255] 범위로 스케일링 한 다음 부호없는 8 비트 정수로 변환
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")

		# 결과 히트 맵을 호출 함수로 반환
		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# 제공된 컬러 맵을 히트 맵에 적용한 다음 입력 이미지에 히트 맵을 오버레이
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

		# 컬러 매핑 된 히트 맵과 출력 된 오버레이 이미지의 2 튜플을 반환합니다.
		return (heatmap, output)
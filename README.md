# Iris_pattern_classification


추후 안드로이드에 넣을 인공 신경망 구현 및 테스트
1. Keras 로 모델을 구현하기 위한 기본 함수들 구현

프로젝트 구조 (모든 신경망 성능 테스트 다음과 같은 구조로 작업)
![image](https://user-images.githubusercontent.com/40026846/115135458-8a6c1180-a053-11eb-882c-edfc585ebb37.png)
- 신경망의 Hyper-parameter 설정
- 학습 시 필요한 Callback 함수 구현
- Data Loader 기능 구현

![image](https://user-images.githubusercontent.com/40026846/115135471-a7084980-a053-11eb-9396-3bb8c2e0c4e8.png)

- Confusion Matrix 및 결과 output 시각화
- PC 버전 체크포인트 파일 변환 구현 (학습시켜 조절된 가중치들을 언제든지 불러와 사용할 수 있도록 학습된 파라미터를 파일 만든 기능)

![image](https://user-images.githubusercontent.com/40026846/115135483-bc7d7380-a053-11eb-86ae-22478719a812.png)

-각 신경망을 Tensorflow(Keras)를 활용하여 구현하고 성능 테스트(텐서보드로도 확인)


----------------------------------------------------------------------------------------------------------

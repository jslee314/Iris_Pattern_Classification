from IrisPattern_vgg.util.constants import *
from IrisPattern_vgg.util.dataloader import DataLoader
from keras import backend as K
from IrisPattern_vgg.util.output import makeoutput, make_dir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CNNUtil.customcallback import CustomCallback

from CNNModels.VGG.model.vgg16v1 import VGG_16
from Examples.Multi_label_color_fashion.pyimagesearch.smallervggnet import SmallerVGGNet
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)

if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

print('# 0) 저장할 파일 생성')
make_dir('./output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
make_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

print('# 1) 데이터셋 생성')
dir = 'D:\Data\iris_pattern\Original2'
x_train, y_train, x_val, y_val, lb = DataLoader.load_data(dir)

print('# 2) 모델 구성(add) & 엮기(compile)')
print('The number of class : ' + str(len(lb.classes_)))
model = SmallerVGGNet.build(width=FLG.WIDTH, height= FLG.HEIGHT,depth=FLG.DEPTH, classes=5, finalAct="sigmoid")
#model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=len(lb.classes_))
#model = MobileNetBuilder.build_mobilenet_v1(input_shape=input_shape, classes=len(lb.classes_))
#model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=len(lb.classes_))

model.summary()

print('#  엮기(compile)')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()

print('#  generator(fit_generator)')
# hist = model.fit(x_train, y_train,
#                  epochs=FLG.EPOCHS,
#                  batch_size=FLG.BATCH_SIZE,
#                  validation_data=(x_val, y_val)
#                  ,callbacks=list_callbacks)

# 클래스 당 1,000 개 미만의 이미지로 작업하는 경우 데이터 확대는 가장 좋은 방법이며 "필수"
train_datagen = ImageDataGenerator(
    rotation_range=5, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	horizontal_flip=True, fill_mode="nearest")

hist = model.fit_generator(
	train_datagen.flow(x_train, y_train, batch_size=FLG.BATCH_SIZE),
	validation_data=(x_val, y_val),
	steps_per_epoch=len(x_train) // FLG.BATCH_SIZE,
	epochs=FLG.EPOCHS, verbose=1)


print('# 4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)


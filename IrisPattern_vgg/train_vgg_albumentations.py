from IrisPattern_vgg.util.constants import *
from IrisPattern_vgg.util.dataloader import DataLoader
from IrisPattern_vgg.util.customcallback import CustomCallback
from keras import backend as K
from IrisPattern_vgg.util.output import makeoutput, make_dir
from IrisPattern_vgg.util.gendataloader import ImageGenerator
from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from tensorflow.keras.losses import categorical_crossentropy
from CNNModels.VGG.model.vgg16v1 import VGG_16
from Examples.Multi_label_color_fashion.pyimagesearch.smallervggnet import SmallerVGGNet
from CNNModels.ResNet.model.resnet import ResnetBuilder

from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder
from albumentations import (Compose,
HorizontalFlip, VerticalFlip, ShiftScaleRotate,
RandomRotate90, Transpose,  RandomSizedCrop, Flip,

RGBShift, HueSaturationValue, ChannelShuffle,
CLAHE, RandomContrast, RandomGamma, RandomBrightness,

JpegCompression, IAAPerspective, OpticalDistortion, GridDistortion, IAAAdditiveGaussianNoise, GaussNoise,
RandomBrightnessContrast,
MotionBlur, MedianBlur,
IAAPiecewiseAffine, IAASharpen, IAAEmboss, OneOf, ToFloat
)

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)

if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

print('# 0) 저장할 파일 생성')
make_dir('./output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')

print('# 2) 모델 구성(add) & 엮기(compile)')
# model, model_size = efficientNet_factory('efficientnet-b1',  load_weights=None, input_shape=(FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH), classes=5)
# model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT,depth= FLG.DEPTH, classes=5, finalAct="sigmoid")
# model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=5)
# model = model = ResnetBuilder.build_resnet_152(input_shape, 5)
# model = MobileNetBuilder.build_mobilenet_v1(input_shape=input_shape, classes=5)
model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=5)

print('#  엮기(compile)')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()

print('# 3) 데이터셋 생성')
data_dir = 'D:\Data\iris_pattern\Original_2'
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ShiftScaleRotate(p=0.8),RandomRotate90(p=0.8), Transpose(p=0.5),
    RandomSizedCrop(min_max_height=(FLG.HEIGHT*2/3, FLG.WIDTH*2/3), height=FLG.HEIGHT, width=FLG.WIDTH,p=0.5),
    RandomContrast(p=0.5), RandomGamma(p=0.5), RandomBrightness(p=0.5)
])

AUGMENTATIONS_TEST = Compose([
    VerticalFlip(p=0.5)
])

dataLoader = {
    'TrainGenerator': ImageGenerator(data_dir + '/train', augmentations=AUGMENTATIONS_TRAIN),
    'ValGenerator': ImageGenerator(data_dir + '/test', augmentations=AUGMENTATIONS_TEST)
}

train_generator = dataLoader.get('TrainGenerator')
val_generator = dataLoader.get('ValGenerator')

print('# 4) 모델 학습  fit / fit_generator(callback)')
hist = model.fit_generator(train_generator,
                           validation_data=val_generator,
                           steps_per_epoch=len(train_generator),
                           validation_steps=len(val_generator),
                           epochs=FLG.EPOCHS,
                           verbose=1, callbacks=list_callbacks)

x_val, y_val, lb = DataLoader.test_load_data(data_dir + '/test')
print('# 4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)


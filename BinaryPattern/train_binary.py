from BinaryPattern.util.constants import *
from BinaryPattern.util.dataloader import DataLoader
from BinaryPattern.util.output import makeoutput, make_dir
from BinaryPattern.util.gendataloader import ImageGenerator
from keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from CNNUtil.customcallback import CustomCallback
from CNNModels.VGG.model.smallervggnet import SmallerVGGNet
from CNNModels.VGG.model.vgg16v1 import VGG_16
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder

from albumentations import (Compose,
HorizontalFlip, VerticalFlip, ShiftScaleRotate,
RandomRotate90, Transpose, RandomSizedCrop, RandomContrast, RandomGamma, RandomBrightness)

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)

if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

make_dir('./output/'+FLG.PROJECT_NAME + '/model_saved')
make_dir('./output/'+FLG.PROJECT_NAME + '/validation_Report')

print('# 2) 모델 구성(add) & 엮기(compile)')
# model, model_size = efficientNet_factory('efficientnet-b1',  load_weights=None, input_shape=(FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH), classes=2)

model = SmallerVGGNet.build(width=FLG.WIDTH, height=FLG.HEIGHT,
                            depth=FLG.DEPTH, classes=2, finalAct="softmax")
# ======  SmallerVGGNet  ====
# Total params: 29,777,794
# Trainable params: 29,774,914
# Non-trainable params: 2,880

# model = SmallerVGGNet.buildv2(input_shape=input_shape, classes=2, finalAct="softmax")
# ======  SmallerVGGNet 2  ====
# `Total params: 7,663,618
# Trainable params: 7,659,970
# Non-trainable params: 3,648
# -----> 120 epoch 에서도 acc 0.3.....

# model = SmallerVGGNet.buildv3(input_shape=input_shape, classes=2, finalAct="softmax")
# ======  SmallerVGGNet 2  ====
# Total params: 7,657,218
# Trainable params: 7,655,106

# Non-trainable params: 2,112
# -----> 120 epoch 에서도 acc 0.4

# model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=2)
# model = model = ResnetBuilder.build_resnet_152(input_shape, 2)
# model = MobileNetBuilder.build_mobilenet_v1(input_shape=input_shape, classes=2)

# model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=2)
# ======  mobilenet_v2  ====
# Total params: 1,029,010
# Trainable params: 1,017,986
# Non-trainable params: 11,024

model.summary()
model.compile(loss=binary_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
list_callbacks = CustomCallback.callback(FLG.PATIENCE, FLG.CKPT_W)

data_dir = FLG.DATA_DIR

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ShiftScaleRotate(p=0.8), RandomRotate90(p=0.8), Transpose(p=0.5),
    RandomContrast(p=0.5), RandomGamma(p=0.5), RandomBrightness(p=0.5)])
AUGMENTATIONS_TEST = Compose([
    VerticalFlip(p=0.5)])

dataLoader = {
    'TrainGenerator': ImageGenerator(data_dir + '/train', augmentations=AUGMENTATIONS_TRAIN),
    'ValGenerator': ImageGenerator(data_dir + '/test', augmentations=AUGMENTATIONS_TEST)}

train_generator = dataLoader.get('TrainGenerator')
val_generator = dataLoader.get('ValGenerator')

print('# 4) 모델 학습  fit / fit_generator(callback)')
hist = model.fit_generator(train_generator,
                           validation_data=val_generator,
                           steps_per_epoch=len(train_generator)*4,
                           validation_steps=len(val_generator)*4,
                           epochs=FLG.EPOCHS,  verbose=1, callbacks=list_callbacks)

x_val, y_val = DataLoader.test_load_data(data_dir + '/test')
makeoutput(x_val, y_val, model, hist)

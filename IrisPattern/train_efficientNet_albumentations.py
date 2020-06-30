from IrisPattern.util.constants import *
from IrisPattern.util.dataloader import DataLoader
from IrisPattern.util.customcallback import CustomCallback
from keras import backend as K
from IrisPattern.util.output import makeoutput, make_dir
from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from IrisPattern.util.gendataloader import ImageGenerator
from tensorflow.keras.losses import categorical_crossentropy

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)

if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

print('# 0) 저장할 파일 생성')
make_dir('./output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
make_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

print('# 2) 모델 구성(add) & 엮기(compile)')
model, model_size = efficientNet_factory('efficientnet-b1',  load_weights=None, input_shape=(FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH), classes=5)
#model.summary()

print('#  엮기(compile)')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()


print('# 3) 데이터셋 생성')
data_dir = 'D:\Data\iris_pattern\Original_2'

dataLoader = {
    'TrainGenerator': ImageGenerator(data_dir + '/train'),
    'ValGenerator': ImageGenerator(data_dir + '/test')
}

train_generator = dataLoader.get('TrainGenerator')
val_generator = dataLoader.get('ValGenerator')

print('# 4) 모델 학습  fit / fit_generator(callback)')
hist = model.fit_generator(train_generator,
                           validation_data=val_generator,
                           steps_per_epoch=len(train_generator),
                           validation_steps=len(val_generator),
                           epochs=FLG.EPOCHS,
                           verbose=1)

# x_train, y_train, x_val, y_val, lb = DataLoader.load_data()
# print('# 4)  모델 학습 결과 저장')
# makeoutput(x_val, y_val, model, hist, lb.classes_)


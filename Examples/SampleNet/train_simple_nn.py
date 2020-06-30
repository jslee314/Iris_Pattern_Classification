# python train_simple_nn.py --dataset_2 train_data --modelsaved output/simple_nn.modelsaved --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
# PyImage/tutorial/train_simple_nn.py

from PyImage.tutorial.constants import *   # window 가 아닌,,..우분투에서 만 가능 ㅜㅜㅜㅜ
from PyImage.tutorial.dataloader import DataLoader
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 1) 데이터셋 생성
dataLoader = DataLoader(data_path=FLG.TRAIN_DATA_PATH)
x_train, y_train, x_val, y_val, lb = dataLoader.train_load(test_ratio=0.25, random_state=42)

# 2) 모델 구성 add
# define the 3072-1024-512-3 architecture using Similartiy
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# 3) 모델 엮기 compile
opt = SGD(lr=FLG.INIT_LR)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# 4) 모델 학습 fit
H = model.fit(x_train, y_train,
			  validation_data=(x_val, y_val),
			  epochs=FLG.EPOCHS,
              batch_size=FLG.BATCH_SIZE)


# 5) 모델 평가하기
predictions = model.predict(x_val, batch_size=FLG.BATCH_SIZE)

print("[INFO] model validation")
print(classification_report(y_val.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

#모델 학습 과정 살펴보기
N = np.arange(0, FLG.EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(FLG.PLOT_SIMPLE)

print("[INFO] save the model saved and label binarizer to disk.")
model.save(FLG.MODEL_SIMPLE)
f = open(FLG.LANEL_BIN_SIMPLE, "wb")
f.write(pickle.dumps(lb))
f.close()
# python train_vgg.py --dataset_2 train_data --modelsaved output/smallvggnet.modelsaved --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png
# PyImage/tutorial/train_vgg.py
import matplotlib
matplotlib.use("Agg") # set the matplotlib backend so figures can be saved in the background
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
from .pyimagesearch.smallvggnet import SmallVGGNet
from .constants import *

data = []
labels = []

# 이미지 경로를 잡고 무작위로 섞음
imagePaths = sorted(list(paths.list_images(FLG.TRAIN_DATA_PATH)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input test_data
for imagePath in imagePaths:
    # load the image, resize it to 64x64 pixels (the required input spatial dimensions of SmallVGGNet), and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)

    # extract the class label from the image path and update the  labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary classification you should use Similartiy' to_categorical function instead as the scikit-learn's LabelBinarizer will not return a vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))


# initialize the modelsaved and optimizer (you'll want to use  binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=FLG.INIT_LR, decay=FLG.INIT_LR / FLG.EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY,
								 batch_size=FLG.BATCH_SIZE),
						validation_data=(testX, testY),
						steps_per_epoch=len(trainX) // FLG.BATCH_SIZE,
						epochs=FLG.EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, FLG.EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(FLG.PLOT_VGG)

# save the modelsaved and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(FLG.MODEL_VGG)
f = open(FLG.LANEL_BIN_VGG, "wb")
f.write(pickle.dumps(lb))
f.close()
from .constants import *
from .showtrain import hist_saved, confusion_matrix_saved
from .saver import ModelSaver
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy

import os

def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)
        print(name, "폴더가 생성되었습니다.")
    else:
        print("해당 폴더가 이미 존재합니다.")

def makeoutput(x_val, y_val, model, hist=None):
    # print('[결과저장 1]   model.evaluate)')
    # loss_and_metrics = model.evaluate(x_val, y_val, batch_size=FLG.BATCH_SIZE)
    # print('loss_and_metrics : ' + str(loss_and_metrics))
    print('[결과저장 2]   모델 학습 과정 그래프 저장: png')
    if hist != None:
        hist_saved(hist)

        print('[결과저장 3]   모델 저장하기: h5, ckpt')
        model.load_weights(FLG.CKPT_W)
        model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
        model.save(FLG.CKPT)


    print('[결과저장 4]   model.predict:  classification_report  & confusion_matrix')
    predictions = model.predict(x_val, batch_size=FLG.BATCH_SIZE)
    f = open(FLG.CONFUSION_MX, 'w')

    print('## classification_report')
    classfi_report = classification_report(y_val.argmax(axis=1),
                                           predictions.argmax(axis=1))
    print(classfi_report)
    f.write(str(classfi_report))



    print('## confusion_matrix')
    confu_mx = confusion_matrix(y_val.argmax(axis=1),
                                predictions.argmax(axis=1))
    f.write(str(confu_mx))
    f.close()

    print('## confusion_matrix plot')
    if hist == None:
        confusion_matrix_saved(confu_mx, False)
    confusion_matrix_saved(confu_mx)






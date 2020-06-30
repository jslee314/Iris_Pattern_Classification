from .constants import *
import itertools
import numpy as np
import matplotlib.pyplot as plt


def hist_saved(hist):
    plt.style.use("ggplot")

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    #plt.show()
    plt.savefig(FLG.PLOT)


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks,  rotation=45)
    plt.yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def confusion_matrix_saved(confu_mx):
    # Compute confusion matrix : confu_mx
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confu_mx,
                          title='Confusion matrix, without normalization')
    plt.savefig(FLG.CONFUSION_MX_PLOT)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confu_mx, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(FLG.CONFUSION_MX_PLOT_NOM)
    plt.show()

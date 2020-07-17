defect_confusion_matrix = [[50, 1],
                           [4, 55]]

lacuna_confusion_matrix = [[38, 3],
                           [6, 34]]

spoke_confusion_matrix = [[13, 1],
                          [7, 18]]

spot_confusion_matrix = [[10, 2],
                         [0, 12]]


matrix = lacuna_confusion_matrix

TN = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]
TP = matrix[1][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)    # recall
specificity = TN / (FP + TN)

print('accuracy : ' + str(accuracy))
print('sensitivity : ' + str(sensitivity))
print('specificity : ' + str(specificity))

from BinaryPattern.util.output import makeoutput, make_dir
from BinaryPattern.util.constants import *
from BinaryPattern.util.dataloader import DataLoader
import keras

h5_path = 'output/smallvgg_defect_resize_200_32/modelsaved/h5/smallvgg_defect_resize_200_32.h5'
data_dir = 'D:/2. data/iris_pattern/Binary/defect_binary'



model = keras.models.load_model(h5_path)

x_val, y_val = DataLoader.test_load_data(data_dir + '/test')

makeoutput(x_val, y_val, model)

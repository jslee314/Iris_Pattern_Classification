import tensorflow as tf

flags = tf.app.flags
FLG = flags.FLAGS

''' train & test data set'''
flags.DEFINE_string('TRAIN_DATA_PATH',
                    'dataset_2/train_data',
                    '입력 데이터 셋 경로')

flags.DEFINE_string('TEST_DATA_PATH',
                    'dataset_2/test_data/dog.jpg',
                    '분류 할 입력 이미지의 경로')


''' train_simple_nn.py '''
flags.DEFINE_string('MODEL_SIMPLE',
                    'output/simple_nn.modelsaved',
                    '훈련 된 모델 출력 경로')

flags.DEFINE_string('LANEL_BIN_SIMPLE',
                    'output/simple_nn_lb.pickle',
                    'label binarizer 경로')

flags.DEFINE_string('PLOT_SIMPLE',
                    'output/simple_nn_plot.png',
                    '출력 경로 정확도 / 손실 그래프')

''' train_vgg.py '''
flags.DEFINE_string('MODEL_VGG',
                    'output/smallvggnet.modelsaved',
                    '훈련 된 모델의 경로')

flags.DEFINE_string('LANEL_BIN_VGG',
                    'output/smallvggnet_lb.pickle',
                    'label binarizer 경로')

flags.DEFINE_string('PLOT_VGG',
                    'output/smallvgg_nn_plot.png',
                    'l출력 경로 정확도 / 손실 그래프')






'''initialize our initial learning rate, of epochs to train for, and batch size'''
flags.DEFINE_integer('WIDTH', 64,
                     'target spatial dimension 너비')
flags.DEFINE_integer('HEIGHT', 64,
                     'target spatial dimension 높이')
flags.DEFINE_integer('DEPTH', 3,
                     'target spatial dimension depth')

flags.DEFINE_float('INIT_LR', 0.01,
                     'INIT_LR')
flags.DEFINE_integer('EPOCHS', 75,
                     '에폭')
flags.DEFINE_integer('BATCH_SIZE', 32,
                     'batch_size')








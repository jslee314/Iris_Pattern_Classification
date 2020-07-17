import tensorflow as tf
FLG = tf.compat.v1.flags.FLAGS

# training hyper parameter 기본 정보
tf.compat.v1.flags.DEFINE_integer('EPOCHS', 200,
                     '같은 모이고사를 몇번 반복할것인지')
tf.compat.v1.flags.DEFINE_integer('BATCH_SIZE', 32,
                     '몇문제 풀어보고 답을 맞출지, 몇개의 샘플로 가중치를 계산하것인지')
tf.compat.v1.flags.DEFINE_integer('PATIENCE', 100,
                     'callback 함수에 있는 EarlyStopping 의 patience, 몇번까지 봐줄것인지')

# 학습 이미지 저장 경로
tf.compat.v1.flags.DEFINE_string('DATA_DIR',
                    'D:/2. data/iris_pattern/Binary/spoke',
                    '불러올 이미지 저장 경로')

# Project 기본 정보
data_model = 'smallervgg_spoke_resize_'

tf.compat.v1.flags.DEFINE_string('PROJECT_NAME',
                    data_model + str(FLG.EPOCHS) + '_' + str(FLG.BATCH_SIZE),
                    '프로젝트 이름 : data + model + epoch + batch size ')



# image 기본 정보 =================================================================================================
tf.compat.v1.flags.DEFINE_integer('WIDTH', 180,
                     'target spatial dimension 너비')
tf.compat.v1.flags.DEFINE_integer('HEIGHT', 180,
                     'target spatial dimension 높이')
tf.compat.v1.flags.DEFINE_integer('DEPTH', 3,
                     'target spatial dimension depth, The CIFAR10 images are RGB.')
dim = 180*180*3
tf.compat.v1.flags.DEFINE_integer('INPUT_DIM', dim,
                     'target spatial dimension 높이, 32*32*3')

# validation result 저장 경로 '''
tf.compat.v1.flags.DEFINE_string('PLOT',
                    './output/'+FLG.PROJECT_NAME +'/validation_Report/plot'+'_'+str(FLG.PROJECT_NAME) + '.png',
                    'plot png 파일 저장할 경로')
tf.compat.v1.flags.DEFINE_string('CONFUSION_MX_PLOT',
                    './output/'+FLG.PROJECT_NAME +'/validation_Report/' + str(FLG.PROJECT_NAME) + '.png',
                    'confusion_matrix 저장 경로')

tf.compat.v1.flags.DEFINE_string('CONFUSION_MX_PLOT_NOM',
                    './output/'+FLG.PROJECT_NAME +'/validation_Report/' + str(FLG.PROJECT_NAME) + '_normal.png',
                    'confusion_matrix 저장 경로')
tf.compat.v1.flags.DEFINE_string('CONFUSION_MX',
                    './output/'+FLG.PROJECT_NAME +'/validation_Report/' + str(FLG.PROJECT_NAME) + '.txt',
                    'confusion_matrix 저장 경로')

# model 저장 경로
tf.compat.v1.flags.DEFINE_string('CKPT',
                    './output/'+FLG.PROJECT_NAME +'/model_saved/' + FLG.PROJECT_NAME+'.h5',
                    '.ckpt 형태로 저장할 경로 : for tensorflow')
tf.compat.v1.flags.DEFINE_string('CKPT_W',
                    './output/'+FLG.PROJECT_NAME +'/model_saved/' + FLG.PROJECT_NAME+'_weight'+'.h5',
                    '.ckpt 형태로 저장할 경로 : for tensorflow')

##
tf.compat.v1.flags.DEFINE_string('TENSORBOARD',
                    './output/'+FLG.PROJECT_NAME +'/tensorboard',
                    'tensorboard 저장할 경로')
tf.compat.v1.flags.DEFINE_string('CSV_LOGGER',
                    './output/'+FLG.PROJECT_NAME +'/validation_Report/'+FLG.PROJECT_NAME +'.csv',
                    'CSV_LOGGER 저장할 경로')
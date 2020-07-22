import tensorflow as tf


# h5_path = './output/binary_smallVGG_defect_x2_400_32.h5'
# tfLite_path = './output/smallVGG_defect.tflite'

# h5_path     = './output/binary_smallVGG_lacuna_x4_400_32/modelsaved/ckpt/binary_smallVGG_lacuna_x4_400_32.h5'
# tfLite_path = './output/smallVGG_lacuna.tflite'
#
# h5_path     = './output/binary_smallVGG_spoke_x4_400_32/modelsaved/ckpt/binary_smallVGG_spoke_x4_400_32.h5'
# tfLite_path = './output/smallVGG_spoke.tflite'
#
# h5_path     = './output/binary_smallVGG_spot_x4_400_16/modelsaved/ckpt/binary_smallVGG_spot_x4_400_16.h5'
# tfLite_path = './output/smallVGG_spot.tflite'

# h5_path =              'util/a_spot_x4_200_16.h5'

# h5_path = './output/mobileNetV2_defect_padding200_16/modelsaved/h5/mobileNetV2_defect_padding200_16.h5'
h5_path = './output/smallervgg_0721_1500_5_200/model_saved/smallervgg_0721_1500_5_200.h5'
tfLite_path = './output/smallervgg_0721_1500_5_200/model_saved/smallervgg_0721_1500_5_200.tflite'

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_path)
flat_data = converter.convert()

with open(tfLite_path, 'wb') as f:
    f.write(flat_data)

# h5_path =             'util/a_lacuna_x4_300_32.h5'
# h5_weights_path = 'util/a_lacuna_x4_300_32_weight.h5'
# model = tf.keras.models.load_model(h5_path)
# model.save_weights(h5_weights_path)
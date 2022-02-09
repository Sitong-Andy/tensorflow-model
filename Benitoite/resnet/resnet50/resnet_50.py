from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    ZeroPadding2D,
    MaxPool2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf


def conv_block(input_tensor, filters, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    name_base = block + "_conv_block_"

    x = Conv2D(filters1, (1, 1), strides, name=name_base + "conv_1")(input_tensor)
    x = BatchNormalization(name=name_base + "bn_1")(x)
    x = Activation("relu", name=name_base + "relu_1")(x)

    x = Conv2D(filters2, (3, 3), padding="same", name=name_base + "conv_2")(x)
    x = BatchNormalization(name=name_base + "bn_2")(x)
    x = Activation("relu", name=name_base + "relu_2")(x)

    x = Conv2D(filters3, (1, 1), padding="valid", name=name_base + "conv_3")(x)
    x = BatchNormalization(name=name_base + "bn_3")(x)

    sc = Conv2D(filters3, (1, 1), strides, name=name_base + "conv_4")(input_tensor)
    sc = BatchNormalization(name=name_base + "bn_4")(sc)

    x = Add(name=name_base + "add_1")([x, sc])
    x = Activation("relu", name=name_base + "relu_3")(x)

    return x


def identity_block(input_tensor, filters, block):
    filters1, filters2, filters3 = filters
    name_base = block + "_identity_block_"

    x = Conv2D(filters1, (1, 1), name=name_base + "conv_1")(input_tensor)
    x = BatchNormalization(name=name_base + "bn_1")(x)
    x = Activation("relu", name=name_base + "relu_1")(x)

    x = Conv2D(filters2, (3, 3), padding="same", name=name_base + "conv_2")(x)
    x = BatchNormalization(name=name_base + "bn_2")(x)
    x = Activation("relu", name=name_base + "relu_2")(x)

    x = Conv2D(filters3, (1, 1), name=name_base + "conv_3")(x)
    x = BatchNormalization(name=name_base + "bn_3")(x)
    x = Add(name=name_base + "add_1")([input_tensor, x])
    x = Activation("relu", name=name_base + "relu_3")(x)

    return x


def resnet_50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

    x = conv_block(x, filters[0], block="2a", strides=(1, 1))
    x = identity_block(x, filters[0], block="2b")
    x = identity_block(x, filters[0], block="2c")

    x = conv_block(x, filters[1], block="3a")
    x = identity_block(x, filters[1], block="3b")
    x = identity_block(x, filters[1], block="3c")
    x = identity_block(x, filters[1], block="3d")

    x = conv_block(x, filters[2], block="4a")
    x = identity_block(x, filters[2], block="4b")
    x = identity_block(x, filters[2], block="4c")
    x = identity_block(x, filters[2], block="4d")
    x = identity_block(x, filters[2], block="4e")
    x = identity_block(x, filters[2], block="4f")

    x = conv_block(x, filters[3], block="5a")
    x = identity_block(x, filters[3], block="5b")
    x = identity_block(x, filters[3], block="5c")

    x = AveragePooling2D((7, 7), name="avg_pool")(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax", name="fc1000")(x)

    model = Model(img_input, x, name="resnet_block")

    return model


def representative_dataset_gen():
    for _ in range(10):
        input_data = np.array(
            np.random.random_sample((1, 224, 224, 3)),
            dtype=np.float32,
        )
        yield [input_data]


def quant_resnet50(shape):
    converter = tf.lite.TFLiteConverter.from_keras_model(resnet_50(input_shape=shape))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter.convert()
    tflite_model_name = "./resnet50.tflite"
    open(tflite_model_name, "wb").write(tflite_quant_model)
    print(tflite_model_name)

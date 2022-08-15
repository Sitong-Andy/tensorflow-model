from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    concatenate,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Dense,
)


def conv2d_bn(x, nb_filter, nb_row, nb_col, padding="same", strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        nb_filter,
        (nb_row, nb_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
    )(x)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation("relu")(x)
    return x


def stem_block(input_tensor):
    x = conv2d_bn(
        input_tensor, 32, 3, 3, strides=(2, 2), padding="valid", name="stem_block_conv1"
    )
    x = conv2d_bn(x, 32, 3, 3, padding="valid", name="stem_block_conv2")
    x = conv2d_bn(x, 64, 3, 3, padding="valid", name="stem_block_conv3")
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding="valid", name="stem_block_conv4")
    x = conv2d_bn(x, 192, 3, 3, padding="valid", name="stem_block_conv5")
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x


def InceptionA(input_tensor, name):
    conv1x1 = conv2d_bn(input_tensor, 64, 1, 1)
    conv5x5 = conv2d_bn(input_tensor, 48, 1, 1)
    conv5x5 = conv2d_bn(conv5x5, 96, 5, 5)
    conv3x3 = conv2d_bn(input_tensor, 64, 1, 1)
    conv3x3 = conv2d_bn(conv3x3, 96, 3, 3)
    conv3x3 = conv2d_bn(conv3x3, 96, 3, 3)
    pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input_tensor)
    pool = conv2d_bn(pool, 64, 1, 1)
    x = concatenate([conv1x1, conv5x5, conv3x3, pool], axis=3, name="mixed0_" + name)
    return x


def InceptionB(input_tensor, name):
    conv3x3 = conv2d_bn(input_tensor, 384, 3, 3, strides=(2, 2), padding="valid")
    conv3x3dbl = conv2d_bn(input_tensor, 64, 1, 1)
    conv3x3dbl = conv2d_bn(conv3x3dbl, 96, 3, 3)
    conv3x3dbl = conv2d_bn(conv3x3dbl, 96, 3, 3, strides=(2, 2), padding="valid")
    pool = MaxPooling2D((3, 3), strides=(2, 2))(input_tensor)
    x = concatenate([conv3x3, conv3x3dbl, pool], axis=3, name="mixed1_" + name)
    return x


def InceptionC(input_tensor, name):
    conv1x1 = conv2d_bn(input_tensor, 192, 1, 1)

    conv7X7 = conv2d_bn(input_tensor, 128, 1, 1)
    conv7X7 = conv2d_bn(conv7X7, 128, 1, 7)
    conv7X7 = conv2d_bn(conv7X7, 192, 7, 1)

    conv7x7dbl = conv2d_bn(input_tensor, 128, 1, 1)
    conv7x7dbl = conv2d_bn(conv7x7dbl, 128, 7, 1)
    conv7x7dbl = conv2d_bn(conv7x7dbl, 128, 1, 7)
    conv7x7dbl = conv2d_bn(conv7x7dbl, 128, 7, 1)
    conv7x7dbl = conv2d_bn(conv7x7dbl, 192, 1, 7)

    pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input_tensor)
    pool = conv2d_bn(pool, 192, 1, 1)
    x = concatenate([conv1x1, conv7X7, conv7x7dbl, pool], axis=3, name="mixed2_" + name)
    return x


def InceptionD(input_tensor, name):
    conv3x3 = conv2d_bn(input_tensor, 192, 1, 1)
    conv3x3 = conv2d_bn(input_tensor, 320, 3, 3, strides=(2, 2), padding="valid")

    conv7x7 = conv2d_bn(input_tensor, 192, 1, 1)
    conv7x7 = conv2d_bn(conv7x7, 192, 1, 7)
    conv7x7 = conv2d_bn(conv7x7, 192, 7, 1)
    conv7x7 = conv2d_bn(conv7x7, 192, 3, 3, strides=(2, 2), padding="valid")

    pool = MaxPooling2D((3, 3), strides=(2, 2))(input_tensor)
    x = concatenate([conv3x3, conv7x7, pool], axis=3, name="mixed3_" + name)
    return x


def InceptionE(input_tensor, name):
    branch1x1 = conv2d_bn(input_tensor, 320, 1, 1)

    conv3x3 = conv2d_bn(input_tensor, 384, 1, 1)
    conv3x3_1 = conv2d_bn(conv3x3, 384, 1, 3)
    conv3x3_2 = conv2d_bn(conv3x3, 384, 3, 1)
    conv3x3 = concatenate([conv3x3_1, conv3x3_2], axis=3)

    conv3x3dbl = conv2d_bn(input_tensor, 448, 1, 1)
    conv3x3dbl = conv2d_bn(conv3x3dbl, 384, 3, 3)
    conv3x3dbl_1 = conv2d_bn(conv3x3dbl, 384, 1, 3)
    conv3x3dbl_2 = conv2d_bn(conv3x3dbl, 384, 3, 1)
    conv3x3dbl = concatenate([conv3x3dbl_1, conv3x3dbl_2], axis=3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input_tensor)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, conv3x3, conv3x3dbl, branch_pool], axis=3, name="mixed4_" + name
    )
    return x


def InceptionV3(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)

    x = stem_block(img_input)

    x = InceptionA(x, name="a1")
    x = InceptionA(x, name="a2")
    x = InceptionA(x, name="a3")

    x = InceptionB(x, name="b1")

    x = InceptionC(x, name="c1")
    x = InceptionC(x, name="c2")
    x = InceptionC(x, name="c3")
    x = InceptionC(x, name="c4")

    x = InceptionD(x, name="d1")

    x = InceptionE(x, name="e1")
    x = InceptionE(x, name="e2")

    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)

    model = Model(img_input, x, name="inception_v3")
    return model

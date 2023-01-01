from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
    Softmax,
)
from tensorflow.keras.models import Model


def VGG16(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(
        input_tensor
    )
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu", name="fc_1")(x)
    x = Dense(4096, activation="relu", name="fc_2")(x)
    output_tensor = Dense(1000, activation="softmax", name="predictions")(x)

    model = Model(input_tensor, output_tensor)
    return model

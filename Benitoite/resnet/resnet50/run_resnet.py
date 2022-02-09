from resnet_50 import resnet_50
import matplotlib.pyplot as plt
import os, PIL
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)
from tensorflow import keras
from tensorflow.keras import layers, models
import pathlib
import argparse


def process_data(data_dir):
    data_dir = pathlib.Path(data_dir)

    batch_size = 8
    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def train_model(model):
    dir = "../../dataset/plant-seedlings-classification/train/"
    train_ds, val_ds = process_data(dir)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-7)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    epochs = 10
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("./resnet_50.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet_50")
    parser.add_argument("--summary", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--quant", type=bool, default=False)

    args = parser.parse_args()
    shape = [224, 224, 3]
    model = resnet_50(shape)

    if args.summary:
        model.summary()

    if args.train:
        train_model(model)

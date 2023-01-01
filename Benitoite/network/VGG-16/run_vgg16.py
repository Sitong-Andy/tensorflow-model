import argparse
import pathlib
from tensorflow import keras
from vgg16 import VGG16
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

batch_size = 8
img_height = 224
img_width = 224
epochs = 20


def process_data(data_dir):
    print("Processing data...")
    data_dir = pathlib.Path(data_dir)
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
    class_names = train_ds.class_names
    print(class_names)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    return train_ds, val_ds, class_names


def train_model(model, train_ds, val_ds):
    print("Training...")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("./vgg16.h5")
    return history


def predict(val_ds, class_names):
    print("Predicting...")
    new_model = keras.models.load_model("./vgg16.h5")
    plt.figure(figsize=(10, 5))
    plt.suptitle("Predict")

    for images, labels in val_ds.take(1):
        print(len(images))
        for i in range(8):
            ax = plt.subplot(2, 4, i + 1)

            plt.imshow(images[i].numpy().astype("uint8"))

            img_array = tf.expand_dims(images[i], 0)

            predictions = new_model.predict(img_array)
            plt.title(class_names[np.argmax(predictions)])

            plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg_16")
    parser.add_argument("--summary", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)

    args = parser.parse_args()
    model = args.model
    summary = args.summary
    train = args.train
    test = args.test

    dir = "/home/sitong/github/dataset/hzw_photos/"
    train_ds, val_ds, class_names = process_data(dir)
    model = VGG16(1000, [img_height, img_width, 3])

    if summary:
        model.summary()
    if train:
        history = train_model(model, train_ds, val_ds)
        # evaluate(history, epochs)
    if test:
        predict(val_ds, class_names)

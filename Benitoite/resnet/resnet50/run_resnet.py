import argparse
import pathlib
from tensorflow import keras
from resnet_50 import resnet_50
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(1)
tf.random.set_seed(1)

epochs = 15
batch_size = 8
img_height = 224
img_width = 224


def set_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(
            gpus[0], True)
        tf.config.set_visible_devices([gpus[0]], "GPU")


def process_data(data_dir):
    print("Processing data...")
    data_dir = pathlib.Path(data_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    print(class_names)
    return train_ds, val_ds, class_names


def train_model(model, train_ds, val_ds):
    print("Training...")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("./resnet_50.h5")
    return history


def predict(val_ds, class_names):
    print("Predicting...")
    new_model = keras.models.load_model('./resnet_50.h5')
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


def evaluate(history, epochs):
    print("Evaluating...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.suptitle("Evaluate")

    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--model", type=str, default="resnet_50")
    parser.add_argument("--summary", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)

    args = parser.parse_args()
    gpu = args.gpu
    model = args.model
    summary = args.summary
    train = args.train
    test = args.test

    model = resnet_50([img_height, img_width, batch_size])
    dir = "/home/sitong/github/tensorflow-model/Benitoite/dataset/plant-seedlings-classification/train/"
    train_ds, val_ds, class_names = process_data(dir)

    if gpu:
        set_gpu()
    if summary:
        model.summary()
    if train:
        history = train_model(model, train_ds, val_ds)
        evaluate(history, epochs)
    if test:
        predict(val_ds, class_names)

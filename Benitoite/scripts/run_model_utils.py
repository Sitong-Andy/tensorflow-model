import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from uuid import uuid4


class utils:
    def __init__(self, model, input_shape, batch_size=1):
        self.model = model
        self.input_shape = input_shape
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.batch_size = batch_size

    def set_gpu(self):
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            # tf.config.experimental.set_memory_growth(gpus[0], True)  # 设置GPU显存用量按需使用
            tf.config.set_visible_devices([gpus[0]], "GPU")

    def short_uuid(self):
        valid_ascii = [[48, 57], [65, 90], [97, 122]]
        uuidChars = []
        for i in valid_ascii:
            for j in range(i[0], i[1] + 1):
                uuidChars.append(chr(j))
        uuid = str(uuid4()).replace("-", "")
        result = ""
        for i in range(0, 8):
            sub = uuid[i * 4 : i * 4 + 4]
            x = int(sub, 16)
            result += uuidChars[x % 0x3E]
        return result

    def gen_ds(self, dir, ds_type):
        if ds_type == "train":
            subset = "training"
        elif ds_type == "val":
            subset = "validation"
        else:
            raise ValueError("ds_type must be train or val")
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset=subset,
            seed=123,
            image_size=(self.height, self.width),
            batch_size=self.batch_size,
        )
        return ds

    def shuffle_ds(self, ds):
        ds = (
            ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return ds

    def compile_model(self, learning_rate=1e-5):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def train_model(self, train_ds, val_ds, epochs=15):
        print("Training...")
        save_path = "./" + self.model.name + "_" + self.short_uuid() + ".h5"
        start_time = time.time()
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        self.model.save(save_path)
        end_time = time.time()
        print("Training time: {}s".format(end_time - start_time))
        return history

    def eval(self, history, epochs):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(epochs)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.suptitle("InceptionV3")

        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    def save_model(self, save_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        open(save_path, "wb").write(tflite_model)

    def representative_dataset_gen(self):
        input_set = []
        for _ in range(1):
            input_data = np.array(
                np.random.random_sample([1, self.height, self.width, self.channels]),
                dtype=np.float32,
            )
            input_set.append(input_data)
        yield input_set

    def quantize_model(self, save_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset_gen
        tflite_model = converter.convert()
        open(save_path, "wb").write(tflite_model)

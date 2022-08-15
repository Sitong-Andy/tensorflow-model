#!/usr/bin/env python3
from random import shuffle
import sys

sys.path.append("../../")
from inception_v3_model import InceptionV3
import argparse
from scripts.run_model_utils import utils

batch_size = 8
img_height = 224
img_width = 224
img_channel = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="inception_v3")
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Print model summary"
    )
    parser.add_argument(
        "--data_set",
        "-ds",
        type=str,
        required=True,
        help="Please input the data set path",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument(
        "--gen_model", "-gm", action="store_true", help="Generate model"
    )

    args = parser.parse_args()
    model = args.model
    summary = args.summary
    data_set = args.data_set
    train = args.train
    test = args.test
    gen_model = args.gen_model

    model = InceptionV3([img_height, img_width, 3], 12)

    run_utils = utils(model, [img_height, img_width, img_channel], batch_size)

    run_utils.set_gpu()

    if gen_model:
        run_utils.save_model("inception_v3.tflite")
        run_utils.quantize_model("inception_v3_quant.tflite")
        exit()

    if summary:
        model.summary()

    if train:
        train_ds = run_utils.gen_ds(data_set, "train")
        val_ds = run_utils.gen_ds(data_set, "val")
        shuffle_train_ds = run_utils.shuffle_ds(train_ds)
        shuffle_val_ds = run_utils.shuffle_ds(val_ds)
        run_utils.compile_model()
        history = run_utils.train_model(shuffle_train_ds, shuffle_val_ds, epochs=20)
        run_utils.eval(history, epochs=20)

    if test:
        pass

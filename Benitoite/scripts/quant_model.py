import argparse
import tensorflow as tf
import numpy as np
from Benitoite.network.GRU.gru_cell import GRU

input_shape = []
quant_bit = 8
en_quant = 1


def representative_dataset_gen():
    for _ in range(10):
        input_data_set = []
        for i in input_shape:
            input_data = np.array(np.random.random_sample(i), dtype=np.float32)
            input_data_set.append(input_data)
        yield input_data_set


def quant_model(model, model_name):
    for shape in model.input_shape:
        input_shape.append(shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if en_quant:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        if quant_bit == 8:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            model_name += "-8bit"
        elif quant_bit == 16:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            model_name += "-16bit"
        converter.experimental_new_quantizer = False
        model_name += "-quant"
    tflite_quant_model = converter.convert()
    tflite_model_path = "./" + model_name + ".tflite"
    open(tflite_model_path, "wb").write(tflite_quant_model)
    print(tflite_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="", required=True)
    parser.add_argument("--en_quant", "-q", type=int, default=1)
    parser.add_argument("--quant_bit", "-qb", type=int, default=8)
    parser.add_argument("--summary", "-s", type=bool, default=False)

    args = parser.parse_args()
    gru = GRU(26, 26)
    model = gru.model()
    if args.model == "gru_cell":
        gru = GRU(26, 26)
        model = gru.model()

    model_name = args.model
    en_quant = args.en_quant
    quant_bit = args.quant_bit
    summary = args.summary

    if summary:
        model.summary()

    quant_model(model, model_name)

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras
import numpy as np

emb_dim = 26
hidden_dim = 26

class GRU:
    def __init__(self, emb_dim=26, hidden_dim=26):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # reset gate
        self.Wr = tf.random.uniform([self.emb_dim, self.hidden_dim])
        self.Ur = tf.random.uniform([self.hidden_dim, self.hidden_dim])
        self.br = tf.random.uniform([self.hidden_dim])

        # update gate
        self.Wz = tf.random.uniform([self.emb_dim, self.hidden_dim])
        self.Uz = tf.random.uniform([self.hidden_dim, self.hidden_dim])
        self.bz = tf.random.uniform([self.hidden_dim])

        # new Memory Cell
        self.Wh = tf.random.uniform([self.emb_dim, self.hidden_dim])
        self.Uh = tf.random.uniform([self.hidden_dim, self.hidden_dim])
        self.bh = tf.random.uniform([self.hidden_dim])

    def reset_gate(self, x, h):
        x_r = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(x)
        h_r = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(h)
        reset = tf.sigmoid(tf.add(x_r, h_r))
        return reset

    def update_gate(self, x, h):
        x_u = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(x)
        h_u = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(h)
        update = tf.sigmoid(tf.add(x_u, h_u))
        return update

    def new_memory_cell(self, x, h, r):
        x_n = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(x)
        h = r * h
        h_n = tf.keras.layers.Dense(self.hidden_dim, activation=None, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')(h)
        gate_update = tf.tanh(tf.add(x_n, h_n))
        return gate_update

    def forward(self, x, h):
        z = self.update_gate(x, h)
        r = self.reset_gate(x, h)
        n = self.new_memory_cell(x, h, r)
        sub = 1 - z
        mul1 = KL.Multiply()([sub, n])
        mul2 = KL.Multiply()([z, h])
        current_hidden_state = KL.Add()([mul1, mul2])
        return current_hidden_state


def representative_dataset_gen():
    for _ in range(8):
        # Get sample input data as a numpy array in a method of your choosing.
        input_data_1 = np.array(np.random.random_sample((1, emb_dim)), dtype=np.float32)
        input_data_2 = np.array(np.random.random_sample((1, hidden_dim)), dtype=np.float32)
        yield [input_data_1, input_data_2]

gru = GRU(emb_dim, hidden_dim)
i = tensorflow.keras.Input(shape=[emb_dim], batch_size=1)
h = tensorflow.keras.Input(shape=[hidden_dim], batch_size=1)
new_h_t = gru.forward(i, h)
model = Model([i, h], new_h_t)
model.summary()
# x = tf.ones((1, 26))
# x2 = tf.ones((1, 26))
# y = model(x, x2)
# tf.print(x)
# tf.print(y)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
converter.experimental_new_converter = False
converter.experimental_new_quantizer = False
tflite_quant_model = converter.convert()
open("./gru_int8.tflite", "wb").write(tflite_quant_model)

import tensorflow.keras
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow import sigmoid, tanh, add

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
        x_r = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(x)
        h_r = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(h)
        reset = sigmoid(add(x_r, h_r))
        return reset

    def update_gate(self, x, h):
        x_u = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(x)
        h_u = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(h)
        update = sigmoid(add(x_u, h_u))
        return update

    def new_memory_cell(self, x, h, r):
        x_n = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(x)
        h = r * h
        h_n = KL.Dense(self.hidden_dim, activation=None, use_bias=True)(h)
        gate_update = tanh(add(x_n, h_n))
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

    def model(self):
        gru = GRU(emb_dim, hidden_dim)
        i = tensorflow.keras.Input(shape=[emb_dim], batch_size=1)
        h = tensorflow.keras.Input(shape=[hidden_dim], batch_size=1)
        new_h_t = gru.forward(i, h)
        model = Model([i, h], new_h_t)
        return model

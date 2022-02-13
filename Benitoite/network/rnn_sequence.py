from gru_cell import GRU
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as KL


class Sequence:
    def __init__(self, model_type="LSTM"):
        self.model_type = model_type

        if self.model_type == "LSTM":
            # TODO create lstm network
            return 0
        elif self.model_type == "GRU":
            self.rnn1 = GRU(1, 51)
            self.rnn2 = GRU(51, 51)
        else:
            print("RNN cannot support ", self.model_type, " network!!!")
            exit()

        self.dense = KL.Dense(units=1)

    def forward(self, input, future=0):
        outputs = []
        h_t = np.zeros([input.size(0), 51], dtype=np.float32)
        c_t = np.zeros([input.size(0), 51], dtype=np.float32)
        h_t2 = np.zeros([input.size(0), 51], dtype=np.float32)
        c_t2 = np.zeros([input.size(0), 51], dtype=np.float32)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            if self.model_type == "LSTM":
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            elif self.model_type == "GRU":
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.dense(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):
            if self.model_type == "LSTM":
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            else:
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.dense(h_t2)
            outputs += [output]
        outputs = tf.stack(outputs, 1)
        return outputs

import tensorflow as tf
from config import config
import numpy as np


class Attention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.m = tf.keras.layers.Dense(units)
        self.h = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, mt, ht):
        wm = self.m(mt)#kxL
        wh = self.h(ht)#kx1
        wh1 = wh * np.ones([config.batch_size, config.num_steps, config.units])#kxL
        ne = tf.nn.tanh(wm + wh1)
        ne = self.v(ne)
        at = tf.nn.softmax(ne)
        ct = mt * at.T #kx1
        return ct, at


class AttentionLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, num_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.num_units = num_units
        self.vocab_size = vocab_size
        self.value_emb = tf.keras.layers.Embedding(vocab_size['value'], embedding_dims['value'])
        self.type_emb = tf.keras.layers.Embedding(vocab_size['type'], embedding_dims['type'])
        self.lstm = tf.keras.layers.LSTM(self.num_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.flag = tf.Variable(dtype=bool, name="start_flag")
        self.attention = Attention(self.num_units)

    def call(self, inputs, hidden):
        type_vec = self.type_emb(inputs[0])
        value_vec = self.value_emb(inputs[1])
        inputs_vec = tf.concat([type_vec, value_vec], axis=2)
        output, state = self.lstm(inputs_vec, initial_state=hidden)
        context_vector, attention_weights = self.attention(output, state)


        return output, state
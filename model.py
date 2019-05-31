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
        wm = self.m(mt)#bxlxk
        h = tf.expand_dims(ht, 1)#bx1xk
        wh = self.h(h)#bx1xk
        ne = self.v(tf.nn.tanh(wm + wh))
        at = tf.nn.softmax(ne, axis=1)
        ct = mt * at
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
        self.fc = tf.keras.layers.Dense(config.vocab_size['value']+config.vocab_size['type'])

    def call(self, inputs, hidden):
        type_vec = self.type_emb(inputs[0])
        value_vec = self.value_emb(inputs[1])
        inputs_vec = tf.concat([type_vec, value_vec], axis=2)
        output, state = self.lstm(inputs_vec, initial_state=hidden)
        context_vector, attention_weights = self.attention(output, state)
        output = self.fc(context_vector)
        return output, state


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


lstm = AttentionLSTM(config.vocab_size, config.embedding_dims, config.units, config.batch_size)


@tf.function
def train_step(inp, targ, hidden):
    loss = 0

    with tf.GradientTape() as tape:
        output, hidden = lstm(inp, hidden)
        input =






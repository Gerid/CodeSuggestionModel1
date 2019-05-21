import tensorflow as tf
from config import config


class Attention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):




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
        


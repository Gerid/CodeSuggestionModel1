import tensorflow as tf
from config import config
import numpy as np
import time
from data_gen import *

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
        #self.flag = tf.Variable(dtype=bool, name="start_flag")
        self.attention = Attention(self.num_units)
        self.fct = tf.keras.layers.Dense(config.vocab_size['type'])
        self.fcv = tf.keras.layers.Dense(config.vocab_size['value'])

    def call(self, inputs, hidden):
        type_vec = self.type_emb(inputs[0])
        value_vec = self.value_emb(inputs[1])
        inputs_vec = tf.concat([type_vec, value_vec], axis=2)
        output, state = self.lstm(inputs_vec, initial_state=hidden)
        context_vector, attention_weights = self.attention(output, state)
        op_type = self.fct(context_vector)
        op_vec = self.fcv(context_vector)
        return [op_type, op_vec], state

    def initialize_hidden_state(self):
        return tf.zeros((self.num_units, self.num_units))


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


type_tensor, type_tokenizer, value_tensor, value_tokenizer, token = load_dataset()

vocab_size = {'type': len(type_tokenizer.word_index)+1, 'value': len(value_tokenizer.word_index)+1}

BATCH_SIZE = 20

dataset = tf.data.Dataset.from_tensor_slices((type_tensor[:, :-1], value_tensor[:, :-1], type_tensor[:, 1:],
                                              value_tensor[:, 1:]))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, _, example_target_batch, _ = next(iter(dataset))

lstm = AttentionLSTM(vocab_size, config.embedding_dims, config.units, config.batch_size)


def train_step(inp, targ, hidden):
    loss = 0

    with tf.GradientTape() as tape:

        for t in range(1, len(targ[0])):
            output, hidden = lstm(inp, hidden)

            loss += loss_function(targ[:, t], output)

            inp = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = lstm.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


#print(example_input_batch, example_target_batch)

steps_per_epoch = config.num_steps


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    hidden = lstm.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp_type, inp_value, targ_type, targ_value)) in enumerate(dataset.take(steps_per_epoch)):
        inp = [inp_type, inp_value]
        targ = [targ_type, targ_value]
        batch_loss = train_step(inp, targ, hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    #if (epoch + 1) % 2 == 0:
    #  checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))







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
        ct = tf.reduce_sum(ct, axis=1)
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
        output, state_h, state_c = self.lstm(inputs_vec)
        context_vector, attention_weights = self.attention(output, state_h)
        op_type = self.fct(context_vector)
        op_vec = self.fcv(context_vector)
        return [op_type, op_vec], state_h


optimizer = tf.train.AdamOptimizer()
loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    #print(real)
    real = tf.keras.utils.to_categorical(real, num_classes=pred.get_shape()[1])
    real = tf.reshape(real, [-1, pred.get_shape()[1]])
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


type_tensor, type_tokenizer, value_tensor, value_tokenizer, token = load_dataset()

vocab_size = {'type': len(type_tokenizer.word_index)+1, 'value': len(value_tokenizer.word_index)+1}

BATCH_SIZE = 200

dataset = tf.data.Dataset.from_tensor_slices((type_tensor[:, :-1], value_tensor[:, :-1], type_tensor[:, 1:],
                                              value_tensor[:, 1:]))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, _, example_target_batch, _ = next(iter(dataset))

lstm = AttentionLSTM(vocab_size, config.embedding_dims, config.units, config.batch_size)


def train_step(inp, targ, hidden):
    loss = 0

    #print(targ)

    with tf.GradientTape() as tape:

        inp = [tf.expand_dims([token[0][0]] * BATCH_SIZE, 1), tf.expand_dims([token[0][1]] * BATCH_SIZE, 1)]

        for t in range(1, targ[0].get_shape()[1]):
            [output_t, output_v], hidden = lstm(inp, hidden)
            output = [output_t, output_v]
            #print(targ[0].get_shape(), output[0].get_shape())
            ls1 = loss_function(targ[0][:, t], output[0])
            ls2 = loss_function(targ[1][:, t], output[1])
            loss += (ls1 + ls2) / 2
            inp = [tf.expand_dims(targ[0][:, t], 1), tf.expand_dims(targ[1][:, t], 1)]

    batch_loss = (loss / int(targ[0].get_shape()[1]))

    variables = lstm.trainable_variables

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


#print(example_input_batch, example_target_batch)

steps_per_epoch = config.num_steps
batch_sz = config.batch_size
nodes = config.units

EPOCHS = 10

print(token)

for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    hidden = tf.zeros((batch_sz, nodes))

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

    ('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))







import tensorflow as tf
import numpy as np
from model import *
import pickle


lstm = AttentionLSTM(config.vocab_size, config.embedding_dims, config.units, config.batch_size)
optimizer = tf.train.AdamOptimizer()


with open('type_vocab.pkl', 'rb') as f:
    type_vocab = pickle.load(f)

with open('value_vocab.pkl', 'rb') as f:
    value_vocab = pickle.load(f)

print(type_vocab)

BATCH_SIZE = config.batch_size


def equal(a, b):
    if a==b:
        return 1
    else:
        return 0

def evaluate():
    inp = [tf.expand_dims([token[0][0]] * BATCH_SIZE, 1), tf.expand_dims([token[0][1]] * BATCH_SIZE, 1)]
    v_accuracy, t_accuracy = 0, 0
    ind = 0
    for (batch, (inp_type, inp_value, targ_type, targ_value)) in enumerate(eval_data.take(10)):
        inp = [inp_type, inp_value]
        targ = [targ_type, targ_value]
        inp2 = [tf.expand_dims([token[0][0]] * BATCH_SIZE, 1), tf.expand_dims([token[0][1]] * BATCH_SIZE, 1)]
        hidden = tf.zeros((batch_sz, nodes))
        len = targ[0].get_shape()[1]
        print("len", len)
        for t in range(1, len):
            [output_t, output_v], hidden = lstm(inp, hidden)
            #output = [output_t, output_v]
            inp = [tf.expand_dims(targ[0][:, t], 1), tf.expand_dims(targ[1][:, t], 1)]
        type_id = tf.argmax(output_t, axis=1).numpy()
        print("typeID:", type_id)
        value_id = tf.argmax(output_v, axis=1).numpy()
        print("valueID:", value_id)
        types, values = [], []
        for i in range(0, len-1):
            types.append(type_vocab[type_id[i]])
            values.append(value_vocab[value_id[i]])
            print(type_vocab[type_id[i]], ":", value_vocab[value_id[i]])
        t1_accuracy, v1_accuracy = 0, 0
        for i in range(0, len-1):
            print("result:", type_id[i], targ[0][i, -1].numpy())
            t1_accuracy += equal(type_id[i], targ[0][i, -1])
            v1_accuracy += equal(value_id[i], targ[1][i, -1])
        print("accuracy:", t1_accuracy, v1_accuracy)
        t_accuracy += t1_accuracy / int(len)
        v_accuracy += v1_accuracy / int(len)
        ind += 1
    t_accuracy /= ind
    v_accuracy /= ind
    return t_accuracy, v_accuracy

print("restoring trained model......")
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

type_tensor, _, value_tensor, _, token = load_dataset(path=config.eval_path, num_examples=200)

eval_data = tf.data.Dataset.from_tensor_slices((type_tensor[:, :-1], value_tensor[:, :-1], type_tensor[:, 1:],
                                              value_tensor[:, 1:]))
eval_data = eval_data.batch(config.batch_size, drop_remainder=True)

print("the result is: ", evaluate())
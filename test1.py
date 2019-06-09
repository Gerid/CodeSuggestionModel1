import tensorflow as tf
import numpy as np
from model import *

lstm = AttentionLSTM(vocab_size, config.embedding_dims, config.units, config.batch_size)
optimizer = tf.train.AdamOptimizer()

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
        for t in range(1, len):
            [output_t, output_v], hidden = lstm(inp, hidden)
            #output = [output_t, output_v]
            inp = [tf.expand_dims(targ[0][:, t], 1), tf.expand_dims(targ[1][:, t], 1)]
        type_id = tf.argmax(output_t, axis=1).numpy()
        types = [type_tokenizer.index_word[type_id[i]] for i in range(0, len-1)]
        value_id = tf.argmax(output_v[0]).numpy()
        values = [value_tokenizer.index_word[value_id[i]] for i in range(0, len-1)]
        t1_accuracy, v1_accuracy = 0, 0
        for i in range(0, len):
            t1_accuracy += equal(types[i], targ[0][i, -1])
            v1_accuracy += equal(values[i], targ[1][i, -1])
        t_accuracy += t1_accuracy/len
        v_accuracy += v1_accuracy/len
        ind += 1
    t_accuracy /= ind
    v_accuracy /= ind
    return t_accuracy, v_accuracy


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

type_tensor, _, value_tensor, _, token = load_dataset(path=config.eval_path, num_examples=200)
eval_data = tf.data.Dataset.from_tensor_slices((type_tensor[:, :-1], value_tensor[:, :-1], type_tensor[:, 1:],
                                              value_tensor[:, 1:]))
eval_data = eval_data.batch(BATCH_SIZE, drop_remainder=True)
print(type_tokenizer.index_word)

print("the result is: ", evaluate())
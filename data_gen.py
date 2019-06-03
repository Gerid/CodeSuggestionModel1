import json
import tensorflow as tf
from config import config
import sys
import numpy as np

tf.enable_eager_execution()

def create_dataset(path, num_examples):
    value_data = []
    type_data = []
    line_index = 0
    fline_index = 0
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            jsonD = json.loads(line)
            word_count = 0
            if fline_index < num_examples:
                value_data.append([])
                type_data.append([])
                value_data[line_index].append('<Start Of Prog>')
                type_data[line_index].append('<Start Of Prog>')
                word_count += 1
                for j in jsonD:
                    if 'value' in j.keys():
                        value_data[line_index].append(j['value'])
                        type_data[line_index].append(j['type'])
                        word_count += 1
                    if word_count >= 50:
                        word_count = 0
                        line_index += 1
                        value_data.append([])
                        type_data.append([])
                if len(value_data[line_index]) > 0:
                    value_data[line_index].append('<End Of Prog>')
                    type_data[line_index].append('<End Of Prog>')
                    line_index += 1
                fline_index += 1
            else:
                del f
                break
    return type_data, value_data


def tokenize(data, num_words=None):
    data_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words,
        filters='')
    data_tokenizer.fit_on_texts(data)

    tensor = data_tokenizer.texts_to_sequences(data)

    eoftoken = data_tokenizer.word_index["<end of prog>"]

    softoken = data_tokenizer.word_index["<start of prog>"]

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post', value=eoftoken)

    return tensor, data_tokenizer, softoken, eoftoken


def load_dataset(path=config.dataset_path, num_examples=config.num_examples):
    # creating cleaned input, output pairs
    type_data, value_data = create_dataset(path, num_examples)

    type_tensor, type_tokenizer, tsoftoken, teoftoken = tokenize(type_data)
    value_tensor, value_tokenizer, vsoftoken, veoftoken = tokenize(value_data, config.vocab_size['value'])

    token=np.zeros((2, 2))
    token[0][0], token[1][0], token[0][1], token[1][1],  = tsoftoken, teoftoken, vsoftoken, veoftoken

    return type_tensor, type_tokenizer, value_tensor, value_tokenizer, token


#type_tensor, type_tokenizer, value_tensor, value_tokenizer, token = load_dataset()
#
#BATCH_SIZE = 20
#
#dataset = tf.data.Dataset.from_tensor_slices((type_tensor[:, :-1], value_tensor[:, :-1], type_tensor[:, 1:],
#                                              value_tensor[:, 1:]))
#dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#example_input_batch, _, example_target_batch, _ = next(iter(dataset))
#print(example_input_batch, example_target_batch)
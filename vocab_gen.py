import json
import tensorflow as tf
import sys
from config import config

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)


dataset_path = config.dataset_path

value_data = []
max_line = 5 #max lines processed in the dataset

line_index = 0
#used
value_data = []
type_data = []

with open(dataset_path, "r", encoding="utf-8") as f:
    for _, line in enumerate(f):
        value_data.append([])
        type_data.append([])
        jsonD = json.loads(line)
        if line_index < max_line:
            for j in jsonD:
                if 'value' in j.keys():
                    value_data[line_index].append(j['value'])
                    type_data[line_index].append(j['type'])
            line_index += 1
        else:
            del f
            break

#num_words in value_tokenizer for size of value vocab
num_words = config.num_words

value_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
type_tokenizer = tf.keras.preprocessing.text.Tokenizer()


#can be optimized with a generator of strings for memory efficiency
value_tokenizer.fit_on_texts(value_data)
type_tokenizer.fit_on_texts(type_data)

with open('value_dic.json', 'w') as vf:
    json.dump(value_tokenizer.word_index, vf)

with open('type_dic.json', 'w') as f:
    json.dump(type_tokenizer.word_index, f)


sys.exit()
#vocab = sorted(set(type_set))
#value_res = sorted(value_dict.items(), key=lambda item: item[1], reverse=True)
#print(len(value_res))
##print(vocab)
##print(len(vocab))
#print(value_res)
#quit()

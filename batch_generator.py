from config import config
import tensorflow as tf
import json


dataset_path = config.dataset_path
valuedic_path = config.valuedic_path
typedic_path = config.typedic_path


value_data = []
max_line = 5 #max lines processed in the dataset

line_index = 0
#used
value_data = []
type_data = []

value_dic = {}

with open(valuedic_path, "r", encoding="utf-8") as f:
    json1 = json.load(f)
    value_dic = json1


with open(typedic_path, "r", encoding="utf-8") as f:
    json1 = json.load(f)
    type_dic = json1

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

inp_data = [type_data[:-1], value_data[:-1]]
targ_data = [type_data[1:], value_data[1:]]





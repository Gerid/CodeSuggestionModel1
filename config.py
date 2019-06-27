import string
import os
import tensorflow as tf

class ModelConfig(object):
    def __init__(self):
        self.batch_size = 64  # 每一批数据的个数
        self.num_words = 200 #value字典的大小
        self.summary_frequency = 100  # 生成样本的频率
        self.num_steps = 50  # 训练步数
        self.num_nodes = 64  # 隐含层个数
        self.root_dir = os.path.dirname(os.path.abspath('.')) + "/CodeSuggestionModel1/"
        self.valuedic_path = self.root_dir + 'value_dic.json'
        self.typedic_path = self.root_dir + 'type_dic.json'
        self.dataset_path = "/home/cr/fang/dataset/python50k_eval.json"
        self.num_examples = 50000 #for test
        self.embedding_dims = {"type": 100, "value": 1000}
        self.vocab_size = {'type': 147, 'value': 1000}
        self.units = 256
        self.eval_path = "/home/cr/fang/dataset/python50k_eval.json"


config = ModelConfig()

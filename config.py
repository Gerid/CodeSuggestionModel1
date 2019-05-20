import string
import os

class ModelConfig(object):
    def __init__(self):
        self.batch_size = 64  # 每一批数据的个数
        self.num_words = 200 #value字典的大小
        self.summary_frequency = 100  # 生成样本的频率
        self.num_steps = 50  # 训练步数
        self.num_nodes = 64  # 隐含层个数
        self.root_dir = os.path.dirname(os.path.abspath('.')) + "/untitled/"
        self.valuedic_path = self.root_dir + '/value_dic.json'
        self.typedic_path = self.root_dir + '/type_dic.json'
        self.dataset_path = "/home/fly/下载/py150/python100k_train.json"
        self.num_examples = 1 #for test




config = ModelConfig()

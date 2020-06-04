import os
import logging
import time

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# 神经网络参数
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]'
                           '-\033[1;33m %(levelname)s \033[0m:\033[1;36m  %(message)s \033[0m ',
                    )
filehandler = logging.FileHandler(filename=os.path.join(ROOT_DIR, "./data/log/log-{}.log".format(time.time())),mode="w")
fileformater = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d]'
                                 '-%(funcName)s- %(levelname)s: %(message)s ')
filehandler.setFormatter(fileformater)
logging.getLogger().addHandler(filehandler)
# 是否使用gpu
USE_CUDA = False
BATCH_SIZE = 32
NUM_EPOCHS = 100
MAX_VOCAB_SIZE = 50000
LEARNING_RATE = 1e-2
EMBEDDING_SIZE = 512
VocabUrl = os.path.join(ROOT_DIR, "./data/embeddingmodel")
embedding_weight_url = os.path.join(ROOT_DIR, "./data/wordembedding/embedding-512.npy")
SL = 128  # 文章最大长度
model_path = os.path.join(ROOT_DIR, "./data/trained_model")



# 总体参数
class_list = ['数据库', 'java', '大数据', 'ai']
# test = os.path.abspath("./data")
dataset_url = os.path.join(ROOT_DIR,"./data/dataset")
# print(os.getcwd())


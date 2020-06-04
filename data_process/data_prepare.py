import jieba
import pandas as pd
import json
import numpy as np
import random
from collections import Counter
from bs4 import BeautifulSoup

STOPWORD = []
STOPWORD_PATH = "D:\\gitRepo\\textclassifier_blog\\data\\stopwords_zh.txt"



def prepare_from_json_file(path: str, target: str, need_format: bool):
    '''
    将数据转换成合适的csv格式，文章进行文字提取，label提取，并
    :param path: 文件路径
    :param target: 目标文件夹
    :param need_format: 是否需要进行分割
    :return: none
    '''
    print("source path:{}, target path:{}".format(path, target))
    pass


def format_(stopword, blog: str) -> list:
    soup = BeautifulSoup(blog, "lxml")
    text = soup.text
    text = jieba.cut(text)
    text = " ".join([i for i in text if i not in stopword])
    return text



def _load_data():
    pass


if __name__ == '__main__':
    stopword = []
    with open(STOPWORD_PATH) as f:
        stopword = [i.strip() for i in f.readlines()]
    print(format_(stopword,"你好我是中国人"))
    print("测试数据预处理")

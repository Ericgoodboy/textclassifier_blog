import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from settings import dataset_url,class_list

class TextClassifier(object):
    def __init__(self, classifier=SVC(kernel='linear'), vectorizer=TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=12000)):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def fit(self, x, y):
        self.classifier.fit(self.feature(x), y)

    def feature(self, x):
        return self.vectorizer.transform(x)

    def score(self, x, y):
        return self.classifier.score(self.feature(x), y)

    def vector_fit(self, x):
        self.vectorizer.fit(x)

    def predict(self, x):
        '''
        :param x: 输入
        :return: 预测值
        '''
        return self.classifier.predict(self.feature(x))


if __name__ == '__main__':
    # df_train = pd.read_csv(os.path.join(dataset_url, "train.csv")).dropna()
    # df_test = pd.read_csv(os.path.join(dataset_url, "test.csv")).dropna()
    # df_valid = pd.read_csv(os.path.join(dataset_url, "valid.csv")).dropna()
    # model = TextClassifier()
    # model.vector_fit(df_train.body)
    #
    # model.fit(df_train.body, df_train.tag)
    # print(model.score(df_test.body, df_test.tag))
    pass

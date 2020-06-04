from model import init_net
from settings import dataset_url
import os
import pandas as pd
df = pd.read_csv(os.path.join(dataset_url,"test.csv"))
sentence = df["body"][3]
tag = df["tag"][3]
net = init_net()
if __name__ == '__main__':
    print(net.predict(sentence))
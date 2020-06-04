import pandas as pd
from settings import dataset_url,class_list
import os


def split_set(setUrl, prefix="test_"):
    df_t = pd.read_csv(setUrl)
    for i in class_list:
        df_now = df_t[df_t["tag"] == i]
        df_now.to_csv(os.path.join(dataset_url,prefix+i+".csv"),index=None)


if __name__ == '__main__':
    url = os.path.join(dataset_url,"valid.csv")
    split_set(url,prefix="valid_")

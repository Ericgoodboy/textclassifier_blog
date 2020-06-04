import os

from model.neural_network import NetModel
from settings import dataset_url
import random
import numpy as np
import torch
import logging
logger = logging.getLogger()


if __name__ == '__main__':
    random.seed(53113)
    np.random.seed(53113)
    torch.manual_seed(53113)
    torch.cuda.manual_seed(53113)
    torch.cuda.manual_seed_all(53113)
    train = os.path.join(dataset_url, "train.csv")
    test = os.path.join(dataset_url, "test.csv")
    net = NetModel(netname="cnn", train_path=train, test_path=test)
    net.fit()
    # score, loss = net.score(os.path.join(dataset_url, "valid.csv"))
    # logging.info("valid_loss:{:.6f} socre:{}".format(loss,score))


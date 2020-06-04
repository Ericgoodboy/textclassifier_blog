import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import pandas as pd
import os
import pickle as pkl
import logging
from settings import SL, VocabUrl, embedding_weight_url, MAX_VOCAB_SIZE, dataset_url,\
    USE_CUDA, LEARNING_RATE, class_list ,model_path
from  data_process import format_
logger = logging.getLogger()
with open(os.path.join(VocabUrl, "w2i.pkl"), "rb") as f:
    w2i = pkl.load(f)
    logger.debug("word to index loaded")
with open(os.path.join(VocabUrl, "i2w.pkl"), "rb") as f:
    i2w = pkl.load(f)
    logger.debug("index to word loaded")

class FastText(nn.Module):
    def __init__(self, embedding_num, vec_dim, label_size, hidden_size):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(embedding_num, vec_dim)
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.Dropout(.5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size)
        )
    def forward(self, x):
        x = self.embed(x)
        out = self.fc(torch.mean(x, dim=1))
        return out
class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        V = args["embed_num"]
        D = args["embed_dim"]
        C = args["class_num"]
        Ci = 1
        Co = args["kernel_num"]
        Ks = args["kernel_sizes"]
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(torch.from_numpy(np.load(embedding_weight_url)))
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

        self.dropout = nn.Dropout(args["dropout"])
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        return x


def map_sentence(sentence:str):
    sentence = sentence.split()
    if len(sentence) > SL:
        sentence = sentence[:SL]
    else:
        sentence += ["<unk>"]*(SL-len(sentence))
    assert len(sentence) == SL
    encoded_sentence = []
    for i in sentence:
        if i in w2i:
            encoded_sentence.append(w2i[i])
        else:
            encoded_sentence.append(w2i["<unk>"])
    return encoded_sentence


class BlogDataset(tud.Dataset):
    def __init__(self, path: str, class_list : list):
        '''
        初始化函数\n
        :param path:数据集位置
        :param class_list:
        '''
        super(BlogDataset, self).__init__()
        df = pd.read_csv(path, encoding="utf8", index_col="index")
        df = df.dropna()
        datas = [[df["body"].iloc[i], df["tag"].iloc[i], df["title"].iloc[i]] for i in range(df.shape[0])]
        self.datasets = []
        self.labels = class_list
        for i in datas:
            sentence = i[0]
            sentence = map_sentence(sentence)
            self.datasets.append((sentence, self.labels.index(i[1])))

        self.class_num = len(self.labels)
    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.datasets)

    def __getitem__(self, idx):
        '''
        切片位置
        :param idx:位置
        :return: 返回这个位置的值
        '''
        return torch.LongTensor(self.datasets[idx][0]), self.datasets[idx][1]


class NetModel(object):
    def __init__(self,netname="cnn",train_path=None,test_path=None):
        args = {
            "embed_num": MAX_VOCAB_SIZE,
            "embed_dim": 512,
            "class_num": 4,
            "kernel_num": 20,
            "kernel_sizes": [1, 7, 3, 5, 5],
            "dropout": 0.3
        }
        self.netname = netname
        with open(os.path.join(VocabUrl, "w2i.pkl"), "rb") as f:
            self.w2i = pkl.load(f)
        with open(os.path.join(VocabUrl, "i2w.pkl"), "rb") as f:
            self.i2w = pkl.load(f)
        if netname == "cnn":
            self.net = CNN_Text(args)
        elif netname == "fasttest":
            self.net = (args["embed_num"], args["embed_dim"], 4,200)
        else:
            self.net = CNN_Text(args)
        # if netname is not None:
        #     self.load_net(netname)
        self.train_path = train_path
        if test_path is not None:
            self.test = self.load_data(test_path, batch_size=100, shuffle=False,test=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, score=-1,epoch=-1):
        net_name = self.netname+"_"+("default" if score == -1 else str(epoch)+"_"+str(score))+".pth"
        torch.save(self.net.state_dict(), os.path.join(model_path, net_name))

    def load_net(self,net_name="cnn_default.pth"):
        self.net.load_state_dict(torch.load(os.path.join(model_path, net_name)))

    def fit(self, dataset=None):
        dataset = dataset if dataset is not None else self.train_path
        if isinstance(dataset, str):
            dataset = self.load_data(dataset)
        assert isinstance(dataset, tud.DataLoader), "传入的训练语料必须是路径或者 Dataloader"
        self.train(dataset, self.loss_fn, self.optimizer)

    def score(self, test_path=None):
        test = test_path if test_path is not None else self.test
        if isinstance(test, str):
            test = self.load_data(test, batch_size=100, shuffle=False,test=True)
        input_, label = next(iter(test))
        if USE_CUDA:
            input_ = input_.cuda()
            label = label.cuda()
        predict = self.predict(input_)
        loss = self.loss_fn(predict,label)
        _, predict = predict.max(1)
        score = (predict == label)
        score = score.sum()
        # logger.debug(score)
        return score.item(),loss

    def train(self,  dataloader, loss_fn, optimizer, epochs=100):
        min_loss=100
        max_score = 0
        if USE_CUDA:
            self.net = self.net.cuda()
        for epoch in range(epochs):
            for i, (sentences, label) in enumerate(dataloader):
                optimizer.zero_grad()
                if USE_CUDA:
                    sentences = sentences.cuda()
                    label = label.cuda()
                output = self.net(sentences)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
            t_score, t_loss = self.score()
            if t_score>=max_score and t_loss<min_loss:
                max_score=t_score
                min_loss = t_loss
                self.save(t_score,epoch)
            logger.info("epoch: {} , loss:{:.3f} score:{} %".format(epoch, t_loss,t_score))

    @staticmethod
    def load_data(url, batch_size=100, shuffle=True, test=False):
        _data = BlogDataset(url, class_list)

        logger.debug(_data.labels)
        if test:
            batch_size = len(_data)
        dataset = tud.DataLoader(_data, batch_size=batch_size, shuffle=shuffle)
        return dataset

    def predict(self, x):
        tostr = False
        if isinstance(x,str):
            tostr = True
            x = format_([], x)
            x = map_sentence(x)
            x = torch.LongTensor([x])
            print(x.shape)
        with torch.no_grad():
            res = self.net(x)
        # print(res)
        if tostr:
            npa=F.softmax(res,dim=1)[0].data.numpy()*100
            for i,cname in enumerate(class_list):
                print("{} : {:.3f}%".format(cname,npa[i]))
            return class_list[res[0].argmax().item()]
        return res


if __name__ == '__main__':
    # net = NetModel()
    train = os.path.join(dataset_url, "train.csv")
    test = os.path.join(dataset_url, "test.csv")
    net = NetModel(train_path=train, test_path=test)
    net.load_net("cnn_4_99.pth")
    res_test = []
    for i in class_list:
        du = os.path.join(dataset_url, "test_{}.csv".format(i))
        length = pd.read_csv(du).shape[0]
        right, _ = net.score(du)
        res_test.append([i, length, right/length])
        print("{}/{}".format(right,length))
    res_valid = []
    for i in class_list:
        du = os.path.join(dataset_url, "valid_{}.csv".format(i))
        length = pd.read_csv(du).shape[0]
        right, _ = net.score(du)
        res_valid.append([i, length, right / length])
        print("{}/{}".format(right,length))



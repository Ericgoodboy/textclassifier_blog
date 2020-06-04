import matplotlib.pyplot as plt
import os
from settings import ROOT_DIR
plt.rcParams["font.sans-serif"]=["SimHei"] #用来正常显示中文标签
plt.rcParams["axes.unicode_minus"]=False #用来正常显示负号
log_path = os.path.join(ROOT_DIR, "data/log/word-embedding.log")
losses = []


def read_log():
    with open(log_path) as f:
        for i in f.readlines():
            if i.startswith("epoch"):
                loss = float(i.split("loss: ")[1])
                losses.append(loss)


if __name__ == '__main__':
    read_log()
    plt.title('词向量训练loss变化曲线')
    plt.plot(losses)
    plt.show()





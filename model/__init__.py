from .neural_network import NetModel


def init_net(net_name="cnn_4_99.pth"):
    net = NetModel()
    net.load_net(net_name=net_name)
    return net

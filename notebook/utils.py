import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms

# dataset loder
def load_dataset(
        data_set : str,
        batch_size : int,
        shuffle=True,
        num_works=1,
        download=True
    ):
    assert data_set in ['MNIST']
    trans = transforms.ToTensor()
    if data_set == 'MNIST':
        train = torchvision.datasets.MNIST(
            root='./data', train=True, transform=trans, download=download
        )
        test = torchvision.datasets.MNIST(
            root='./data', train=False, transform=trans, download=download
        )
    return (data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_works),
            data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_works))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train_m(
        net : nn.Module,
        train_iter : iter,  # iterator of train set
        updater : torch.optim.Optimizer,
        loss : float,
        num_epochs : int
    ):
    net.train()
    for i in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()  # 梯度清零
            l.mean().backward()  # 误差反向传播
            updater.step()  # 损失函数梯度下降
        print(f'epoch {i+1}, train set loss:{l}')

def evaluate_m(net, test_iter):
    net.eval()
    with torch.no_grad():
        num_all = 0
        num_correct = 0
        for X, y in test_iter:
            t = net(X).argmax(axis=1) == y
            num_all += len(t)
            num_correct += torch.sum(t)
    acc = num_correct / num_all
    print(f'evaluation, accuracy:{acc}')
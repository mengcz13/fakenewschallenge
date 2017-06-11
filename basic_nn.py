import torch
import torch.nn as nn
from torch.autograd import Variable


class BasicNNModel(nn.Module):
    def __init__(self, inputn, title_len, body_len, nhidden, nclass, dropout=0.1):
        super(BasicNNModel, self).__init__()
        self.inputn = inputn
        self.title_len = title_len
        self.body_len = body_len
        self.nhidden = nhidden
        self.nclass = nclass

        self.drop = nn.Dropout(dropout)
        self.avgpool = nn.AvgPool1d(title_len + body_len)
        self.linear = nn.Linear(self.inputn, self.nhidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.nhidden, self.nclass)
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.linear2(
            self.drop((self.tanh((self.linear(self.avgpool(torch.transpose(x, 1, 2)).view(-1, self.inputn))))))))


if __name__ == '__main__':
    nclass = 4
    title_len = 20
    body_len = 100
    hidden = 42
    bsz = 5
    wemb = 50
    input = Variable(torch.rand(bsz, title_len + body_len, wemb))

    bm = BasicNNModel(wemb, title_len, body_len, hidden, nclass)

    print(bm.forward(input))

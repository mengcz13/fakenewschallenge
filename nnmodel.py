# -*- coding: utf-8 -*-
# Main model for classification.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np


class CNNSDModel(nn.Module):
    def __init__(self, input_size, out_channels, kernel_width):
        super(CNNSDModel, self).__init__()
        self.cnn = nn.Conv2d(1, out_channels, kernel_size=(kernel_width, input_size[2]), stride=1, padding=0)
        self.conv_outsize = self._get_conv_outsize(input_size, out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((self.conv_outsize, 1))
        self.outsize = out_channels

    def forward(self, input):
        x = self.relu(self.cnn(input))
        x = self.maxpool(x)
        x = x.view(-1, self.outsize)
        return x

    def _get_conv_outsize(self, input_size, out_channels):
        bs = 1
        input = Variable(torch.rand(bs, *input_size))
        output = self.cnn(input)
        n_size = output.data.view(bs, out_channels, -1).size(2)
        return n_size


class NNModel(nn.Module):
    def __init__(self, title_input_size, title_out_channels, title_kernel_width, body_input_size, body_out_channels,
                 body_kernel_width, dropout, catnum):
        super(NNModel, self).__init__()
        self.titlelen = title_input_size[1]
        self.title_cnn = CNNSDModel(title_input_size, title_out_channels, title_kernel_width)
        self.body_cnn = CNNSDModel(body_input_size, body_out_channels, body_kernel_width)
        self.linear = nn.Linear(title_out_channels + body_out_channels, catnum)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = input.view(-1, 1, input.size(1), input.size(2))
        title_input = input[:, :, :self.titlelen, :]
        body_input = input[:, :, self.titlelen:, :]
        temb = self.title_cnn(title_input)
        bemb = self.body_cnn(body_input)
        self.embedding = (temb, bemb)
        emb = torch.cat((temb, bemb), dim=1)
        return self.softmax(self.linear(self.dropout(emb)))


if __name__ == '__main__':
    maxtitlelen = 40
    maxbodylen = 100
    vecdim = 50
    stancelen = 4
    cnn = NNModel(
        title_input_size=(1, maxtitlelen, vecdim),
        title_out_channels=vecdim,
        title_kernel_width=3,
        body_input_size=(1, maxbodylen, vecdim),
        body_out_channels=vecdim,
        body_kernel_width=5,
        dropout=0.1,
        catnum=stancelen
    )
    print(cnn(Variable(torch.rand((100, 140, 50)))))

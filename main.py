# -*- coding: utf-8 -*-

from word2vec_model import Data
from nnmodel import NNModel
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data',
                        help='path to the data folder, which contains GoogleNews-vectors and training data')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size in each iteration')
    parser.add_argument('--iter', type=int, default=50,
                        help='iteration times')
    parser.add_argument('--cuda', action='store_true',
                        help='use if you want the model to run on CUDA')
    options = parser.parse_args()

    traindata = Data(options.datapath)

    # Build model, optimizer and loss function
    nnmodel = NNModel(title_input_size=(1, traindata.maxtitlelen, traindata.vecdim),
                      title_out_channels=300,
                      title_kernel_width=3,
                      body_input_size=(1, traindata.maxbodylen, traindata.vecdim),
                      body_out_channels=300,
                      body_kernel_width=5,
                      catnum=traindata.stancelen)
    if options.cuda:
        nnmodel = nnmodel.cuda()

    optimizer = optim.Adadelta(nnmodel.parameters(), lr=1e-4, weight_decay=1e-6)
    loss_func = nn.CrossEntropyLoss()

    batch_size = options.batch_size
    for t in range(options.iter):
        # generate a batch
        binput, boutput = traindata.get_batch(batch_size)
        if options.cuda:
            binput, boutput = binput.cuda(), boutput.cuda()

        # do optimization
        optimizer.zero_grad()
        boutput_pred_score = nnmodel(binput)
        loss = loss_func(boutput_pred_score, boutput)
        _, boutput_pred = torch.max(boutput_pred_score, dim=1)
        boutput_pred = boutput_pred.view(-1)
        accuracy = torch.mean(torch.eq(boutput_pred, boutput).float())
        print(t, 'loss =', loss.data[0], 'accu =', accuracy.data[0])
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
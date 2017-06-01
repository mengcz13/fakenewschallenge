# -*- coding: utf-8 -*-

from word2vec_model import Data
from nnmodel import NNModel
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from argparse import ArgumentParser
import shutil
import itchat
import datetime
import sys

itchat.auto_login(enableCmdQR=2, hotReload=True)

original_stdout = sys.stdout


class RedirectedWechatOut(object):
    def write(self, s):
        try:
            itchat.send(('%s' % (s)).rstrip('\r\n'), toUserName='filehelper')
            original_stdout.write(s)
        except:
            pass


redirectedWechatOut = RedirectedWechatOut()
sys.stdout = redirectedWechatOut


def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data',
                        help='path to the data folder, which contains GoogleNews-vectors and training data')
    parser.add_argument('--word2vecmodelfile', type=str, default='GoogleNews-vectors-negative300.bin',
                        help='file name of pre-trained word2vec model')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='ratio of train set in all data')
    parser.add_argument('--remove_stopwords', action='store_true',
                        help='flag for removing stopwords')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size in each iteration')
    parser.add_argument('--epoch', type=int, default=50,
                        help='epoch times')
    parser.add_argument('--cuda', action='store_true',
                        help='use if you want the model to run on CUDA')
    parser.add_argument('--iterpbatch', type=int, default=1,
                        help='iteration time of per batch')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--title_trunc', type=int, default=20,
                        help='truncate title')
    parser.add_argument('--body_trunc', type=int, default=100,
                        help='truncate body')
    options = parser.parse_args()
    print(options)

    traindata = Data(options.datapath, options.word2vecmodelfile, options.train_ratio, options.remove_stopwords,
                     options.title_trunc,
                     options.body_trunc)

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

    optimizer = optim.Adadelta(nnmodel.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    loss_func = nn.CrossEntropyLoss()
    if options.cuda:
        loss_func = loss_func.cuda()

    batch_size = options.batch_size

    best_accu = 0

    save_checkpoint({
        'epoch': -1,
        'cpoch_num': -1,
        'state_dict': nnmodel.state_dict(),
        'best_accu': best_accu,
        'optimizer': optimizer.state_dict()
    }, True)

    starttime = datetime.datetime.now()
    print('Start training at: %s' % starttime)

    for t in range(options.epoch):

        cpoch_num = 0

        while cpoch_num < traindata.trainsize:
            cpoch_starttime = datetime.datetime.now()
            # generate a batch
            binput, boutput = traindata.train_get_batch(batch_size)
            print('A Train Batch is generated!')
            cpoch_num += boutput.size()[0]
            if options.cuda:
                binput, boutput = binput.cuda(), boutput.cuda()
            binput = Variable(binput, requires_grad=True)
            boutput = Variable(boutput, requires_grad=False)
            for t2 in range(options.iterpbatch):
                # do optimization
                optimizer.zero_grad()
                boutput_pred_score = nnmodel(binput)
                loss = loss_func(boutput_pred_score, boutput)
                currtime = datetime.datetime.now()
                usedtime = currtime - cpoch_starttime
                esttime = usedtime * (traindata.trainsize - cpoch_num) / (cpoch_num)
                print('%s epoch=%s, cpoch_num=%s, iterpbatch=%s, loss=%s, usedtime=%s' % (
                    currtime, t, cpoch_num, t2, loss.data[0], usedtime))
                loss.backward()
                optimizer.step()
            _, boutput_pred = torch.max(boutput_pred_score, dim=1)
            boutput_pred = boutput_pred.view(-1)
            batch_accu = torch.mean(torch.eq(boutput_pred, boutput).float())
            currtime = datetime.datetime.now()
            usedtime = currtime - starttime
            esttime = usedtime * (traindata.trainsize - cpoch_num) / (cpoch_num)
            print('epoch=%s, cpoch_num=%s, batch_accu=%s, esttime=%s' % (t, cpoch_num, batch_accu.data[0], esttime))

            if cpoch_num >= traindata.trainsize:
                vpoch_num = 0
                mean_accu = 0
                while vpoch_num < traindata.validatesize:
                    vinput, voutput = traindata.validate_get_batch(batch_size)
                    vpoch_num += voutput.size()[0]
                    if options.cuda:
                        vinput, voutput = vinput.cuda(), voutput.cuda()
                    vinput = Variable(vinput, requires_grad=True)
                    voutput = Variable(voutput, requires_grad=False)
                    voutput_pred_score = nnmodel(vinput)
                    _, voutput_pred = torch.max(voutput_pred_score, dim=1)
                    voutput_pred = voutput_pred.view(-1)
                    accuracy = torch.mean(torch.eq(voutput_pred, voutput).float())
                    mean_accu += (voutput.size()[0] * accuracy.data[0])
                    print('vpoch_num=%s, temp_accuracy=%s' % (vpoch_num, accuracy.data[0]))
                mean_accu = mean_accu / traindata.validatesize
                print('epoch=%s, validation accuracy=%s' % (t, mean_accu))

                isbest = (mean_accu > best_accu)
                save_checkpoint({
                    'epoch': t,
                    'cpoch_num': cpoch_num,
                    'state_dict': nnmodel.state_dict(),
                    'best_accu': best_accu,
                    'accu': mean_accu,
                    'optimizer': optimizer.state_dict()
                }, isbest, filename='checkpoint_%s.pth.tar' % t)
                if isbest:
                    best_accu = mean_accu


def save_checkpoint(state, isbest, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if isbest:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()

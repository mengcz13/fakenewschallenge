# -*- coding: utf-8 -*-

from word2vec_model import Data
from nnmodel import NNModel
from birnn import BiRNNModel
from attentionrnn import GlobalNMTAttentionRNNModel, QAGlobalAttentionRNNModel
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
import random
import os
from sklearn.metrics import confusion_matrix


class RedirectedWechatOut(object):
    def __init__(self, originout):
        self.originout = originout

    def write(self, s):
        try:
            itchat.send(('%s' % (s)).rstrip('\r\n'), toUserName='filehelper')
            self.originout.write(s)
        except:
            pass

class ModelWithEmbedding(nn.Module):
    def __init__(self, wordnum, vecdim, model):
        super(ModelWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(wordnum, vecdim)
        self.model = model

    def forward(self, x):
        return self.model(self.embedding(x))


def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data',
                        help='path to the data folder, which contains GoogleNews-vectors and training data')
    parser.add_argument('--word2vecmodelfile', type=str, default='glove.twitter.27B.50d.txt',
                        help='file name of pre-trained word2vec model')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of train set in all data')
    parser.add_argument('--remove_stopwords', action='store_true',
                        help='flag for removing stopwords')
    parser.add_argument('--batch_size', type=int, default=384,
                        help='batch size in each iteration')
    parser.add_argument('--epoch', type=int, default=50,
                        help='epoch times')
    parser.add_argument('--cuda', action='store_true',
                        help='use if you want the model to run on CUDA')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--title_trunc', type=int, default=40,
                        help='truncate title')
    parser.add_argument('--body_trunc', type=int, default=100,
                        help='truncate body')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--wechat', action='store_true',
                        help='use wechat to notify')
    parser.add_argument('model', type=str,
                        help='type of model')
    parser.add_argument('modelpath', type=str,
                        help='path to save model')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    options = parser.parse_args()
    print(options)

    if options.wechat:
        itchat.auto_login(enableCmdQR=2, hotReload=True)
        original_stdout = sys.stdout
        redirectedWechatOut = RedirectedWechatOut(original_stdout)
        sys.stdout = redirectedWechatOut

    random.seed(options.seed)

    MODELPATH = options.modelpath
    if not os.path.exists(MODELPATH):
        os.mkdir(MODELPATH)

    traindata = Data(options.datapath, options.word2vecmodelfile, options.train_ratio, options.remove_stopwords,
                     options.title_trunc,
                     options.body_trunc)

    # Build model, optimizer and loss function
    models = {
        'cnn': NNModel(
            title_input_size=(1, traindata.maxtitlelen, traindata.vecdim),
            title_out_channels=traindata.vecdim,
            title_kernel_width=3,
            body_input_size=(1, traindata.maxbodylen, traindata.vecdim),
            body_out_channels=traindata.vecdim,
            body_kernel_width=5,
            hiddennum=2 * traindata.vecdim,
            dropout=0.1,
            catnum=traindata.stancelen
        ),
        'rnn_lstm_cond_bi': BiRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='LSTM',
            dropout=0.1,
            conditional=True,
            bidirectional=True,
            on_cuda=options.cuda
        ),
        'rnn_gru_cond_bi': BiRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='GRU',
            dropout=0.1,
            conditional=True,
            bidirectional=True,
            on_cuda=options.cuda
        ),
        'rnn_lstm_nocond_bi': BiRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='LSTM',
            dropout=0.1,
            conditional=False,
            bidirectional=True,
            on_cuda=options.cuda
        ),
        'rnn_gru_nocond_bi': BiRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='GRU',
            dropout=0.1,
            conditional=False,
            bidirectional=True,
            on_cuda=options.cuda
        ),
        'globalnmt_attention_gru_cond_bi': GlobalNMTAttentionRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='GRU',
            dropout=0.1,
            conditional=True,
            bidirectional=True,
            on_cuda=options.cuda
        ),
        'qaglobal_attention_gru_cond_bi': QAGlobalAttentionRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=4 * traindata.vecdim,
            nclass=traindata.stancelen,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            rnntype='GRU',
            dropout=0.1,
            conditional=True,
            bidirectional=True,
            on_cuda=options.cuda
        )
    }

    nnmodel = models[options.model]

    if options.cuda:
        nnmodel = nnmodel.cuda()

    optimizer = getattr(optim, options.optimizer)(nnmodel.parameters(),
                                                  lr=options.lr,
                                                  weight_decay=options.weight_decay)

    loss_func = nn.CrossEntropyLoss()
    if options.cuda:
        loss_func = loss_func.cuda()

    best_score = 0

    starttime = datetime.datetime.now()
    print('Start training at: %s' % starttime)

    for t in range(options.epoch):
        train_mean_loss, train_mean_accu, train_mean_cmatrix, train_score = train_epoch(traindata, nnmodel, optimizer,
                                                                                        loss_func, options, t)
        validate_mean_loss, validate_mean_accu, validate_mean_cmatrix, validate_score = validate_epoch(traindata,
                                                                                                       nnmodel,
                                                                                                       loss_func,
                                                                                                       options, t)

        isbest = (validate_score > best_score)
        save_checkpoint({
            'epoch': t,
            'train_result': (train_mean_loss, train_mean_accu, train_mean_cmatrix, train_score),
            'validate_result': (validate_mean_loss, validate_mean_accu, validate_mean_cmatrix, validate_score),
            'state_dict': nnmodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'options': options
        }, isbest, MODELPATH, filename='checkpoint_%s.pth.tar' % t)
        if isbest:
            best_score = validate_score


def train_epoch(data, model, optimizer, loss_func, options, epoch_t):
    model.train()
    starttime = datetime.datetime.now()
    totalbn = data.trainsize // options.batch_size
    mean_loss = 0
    mean_accu = 0
    totalnum = 0
    cmatrix = 0
    for binput, boutput, bn, _ in data.train_batch(options.batch_size):
        # optimization
        binput, boutput = model.format_data(binput, boutput)
        if options.cuda:
            binput, boutput = binput.cuda(), boutput.cuda()
        optimizer.zero_grad()
        boutput_pred_score = model(binput)
        loss = loss_func(boutput_pred_score, boutput)
        loss.backward()
        optimizer.step()

        absize = boutput.size()[0]
        totalnum += absize
        mean_loss += loss.data[0] * absize
        accu = get_accuracy(boutput_pred_score, boutput)
        mean_accu += accu.data[0] * absize
        cmatrix += get_confusion_matrix(boutput_pred_score, boutput)

        usedtime = datetime.datetime.now() - starttime
        esttime = usedtime * (totalbn - bn + 1) / (bn + 1)
        if bn % 10 == 0:
            print('epoch=%s, batch_num=%s, batch_loss=%s, batch_accu=%s, esttime=%s' % (
                epoch_t, bn, loss.data[0], accu.data[0], esttime))

    mean_accu /= totalnum
    mean_loss /= totalnum
    return mean_loss, mean_accu, cmatrix, get_score(cmatrix)


def validate_epoch(data, model, loss_func, options, epoch_t):
    model.eval()
    mean_loss = 0
    mean_accu = 0
    totalnum = 0
    cmatrix = 0
    for vinput, voutput, bn, _ in data.validate_batch(options.batch_size):
        vinput, voutput = model.format_data(vinput, voutput)
        if options.cuda:
            vinput, voutput = vinput.cuda(), voutput.cuda()
        voutput_pred_score = model(vinput)

        loss = loss_func(voutput_pred_score, voutput)
        mean_loss += (voutput.size()[0] * loss.data[0])

        accuracy = get_accuracy(voutput_pred_score, voutput)
        mean_accu += (voutput.size()[0] * accuracy.data[0])

        totalnum += voutput.size()[0]

        cmatrix += get_confusion_matrix(voutput_pred_score, voutput)

    mean_loss = mean_loss / totalnum
    mean_accu = mean_accu / totalnum
    print('epoch=%s, validation loss=%s, validation accuracy=%s, validation score=%s' %
          (epoch_t, mean_loss, mean_accu, get_score(cmatrix)))
    print(cmatrix)
    return mean_loss, mean_accu, cmatrix, get_score(cmatrix)


def get_accuracy(pred_score, output):
    _, output_pred = torch.max(pred_score, dim=1)
    output_pred = output_pred.view(-1)
    accuracy = torch.mean(torch.eq(output_pred, output).float())
    return accuracy


def get_confusion_matrix(pred_score, output):
    _, output_pred = torch.max(pred_score, dim=1)
    output_pred = output_pred.view(-1)
    confmat = confusion_matrix(output.cpu().data.numpy(), output_pred.cpu().data.numpy(), labels=[0, 1, 2, 3])
    return confmat


def get_score(confmat):
    score = 0
    for t in range(4):
        score += confmat[t][t]
    for t1 in range(3):
        for t2 in range(3):
            if t1 != t2:
                score += 0.25 * confmat[t1][t2]
    return score / np.sum(confmat)


def save_checkpoint(state, isbest, filepath, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(filepath, filename))
    if isbest:
        shutil.copyfile(os.path.join(filepath, filename), os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()

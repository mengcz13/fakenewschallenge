# -*- coding: utf-8 -*-

from word2vec_model import EmbData
from nnmodel import NNModel
from birnn import BiRNNModel
from attentionrnn import GlobalNMTAttentionRNNModel, QAGlobalAttentionRNNModel
from cnn_rnn import CNNRNNModel
from basic_nn import BasicNNModel
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
import pickle
import re


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
    def __init__(self, wordnum, vecdim, model, pretrained_embedding=None, fix_embedding=False):
        super(ModelWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(wordnum, vecdim)
        self.model = model
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            if fix_embedding:
                self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.model(self.embedding(x))

    def format_data(self, xin, yin):
        xformateed = Variable(torch.LongTensor(xin), requires_grad=False)
        yformateed = Variable(torch.LongTensor(yin).view(-1), requires_grad=False)
        return xformateed, yformateed


def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data',
                        help='path to the data folder')
    parser.add_argument('--word2vecmodelfile', type=str, default='glove.twitter.27B.50d.txt',
                        help='file name of pre-trained word2vec model')
    parser.add_argument('--fix_embedding', action='store_true',
                        help='fix pre-trained embedding')
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
    parser.add_argument('--selfemb', type=int, default=None,
                        help='add embedding as a section of network, instead of using pre-trained word2vec')
    parser.add_argument('--resume', action='store_true',
                        help='resume from the latest model and train until reaching the epoch num')
    options = parser.parse_args()
    print(options)

    if options.wechat:
        itchat.auto_login(enableCmdQR=2, hotReload=True)
        original_stdout = sys.stdout
        redirectedWechatOut = RedirectedWechatOut(original_stdout)
        sys.stdout = redirectedWechatOut

    random.seed(options.seed)
    torch.manual_seed(options.seed)
    if options.cuda:
        torch.cuda.manual_seed(options.seed)

    MODELPATH = options.modelpath
    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)

    checkpoint = None
    modelbest = None
    if options.resume:
        checkpoint, modelbest = load_latest_checkpoint(MODELPATH)

    if options.selfemb is not None:
        traindata = EmbData(options.datapath, options.train_ratio, options.remove_stopwords, options.title_trunc,
                            options.body_trunc, selfembedding=True, word2vecmodelfile=None, vecdim=options.selfemb)
    else:
        traindata = EmbData(options.datapath, options.train_ratio, options.remove_stopwords, options.title_trunc,
                            options.body_trunc, selfembedding=False, word2vecmodelfile=options.word2vecmodelfile,
                            vecdim=None)

    # Build model, optimizer and loss function
    models = {
        'basic_nn': BasicNNModel(
            inputn=traindata.vecdim,
            title_len=traindata.maxtitlelen,
            body_len=traindata.maxbodylen,
            nhidden=2*traindata.vecdim,
            nclass=traindata.stancelen,
            dropout=0.1
        ),
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
        ),
        'cnn_rnn': CNNRNNModel(
            titleninput=traindata.vecdim,
            bodyninput=traindata.vecdim,
            nhidden=traindata.vecdim,
            nlayers=2,
            classify_hidden=2 * traindata.vecdim,
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

    if options.selfemb is not None:
        nnmodel = ModelWithEmbedding(traindata.wordnum, traindata.vecdim, models[options.model],
                                     pretrained_embedding=None, fix_embedding=False)
    else:
        nnmodel = ModelWithEmbedding(traindata.wordnum, traindata.vecdim, models[options.model],
                                     pretrained_embedding=traindata.embedding, fix_embedding=options.fix_embedding)

    if options.cuda:
        nnmodel = nnmodel.cuda()

    optimizer = getattr(optim, options.optimizer)(filter(lambda p: p.requires_grad, nnmodel.parameters()),
                                                  lr=options.lr,
                                                  weight_decay=options.weight_decay)

    loss_func = nn.CrossEntropyLoss()
    if options.cuda:
        loss_func = loss_func.cuda()

    start_epoch = 0
    if checkpoint:
        nnmodel.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['running_record']['epoch'] + 1
    best_score = 0
    if modelbest:
        best_score = modelbest['running_record']['validate_result'][3]

    starttime = datetime.datetime.now()
    print('Start training at: %s' % starttime)

    running_records = []
    if options.resume and os.path.exists(os.path.join(MODELPATH, 'records.pkl')):
        with open(os.path.join(MODELPATH, 'records.pkl'), 'rb') as f:
            running_records = pickle.load(f)

    for t in range(start_epoch, options.epoch):
        train_mean_loss, train_mean_accu, train_mean_cmatrix, train_score = train_epoch(traindata, nnmodel, optimizer,
                                                                                        loss_func, options, t)
        validate_mean_loss, validate_mean_accu, validate_mean_cmatrix, validate_score = validate_epoch(traindata,
                                                                                                       nnmodel,
                                                                                                       loss_func,
                                                                                                       options, t)

        isbest = (validate_score > best_score)
        running_record = {
            'epoch': t,
            'train_result': (train_mean_loss, train_mean_accu, train_mean_cmatrix, train_score),
            'validate_result': (validate_mean_loss, validate_mean_accu, validate_mean_cmatrix, validate_score),
            'options': options
        }
        running_records.append(running_record)
        save_checkpoint({
            'running_record': running_record,
            'state_dict': nnmodel.state_dict(),
            'optimizer': optimizer.state_dict()
        }, isbest, MODELPATH, filename='checkpoint_%s.pth.tar' % t)
        if isbest:
            best_score = validate_score

    with open(os.path.join(MODELPATH, 'records.pkl'), 'wb+') as f:
        pickle.dump(running_records, f)


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
    return (np.sum(confmat[:3, :3]) * 0.25 + confmat[3][3] * 0.25 + (
        confmat[0][0] + confmat[1][1] + confmat[2][2]) * 0.75) / (np.sum(confmat[:3, :]) + np.sum(confmat[3, :]) * 0.25)


def save_checkpoint(state, isbest, filepath, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(filepath, filename))
    if isbest:
        shutil.copyfile(os.path.join(filepath, filename), os.path.join(filepath, 'model_best.pth.tar'))


def load_latest_checkpoint(modelpath):
    try:
        max_epoch = max([int(m.group(1)) for m in filter(lambda x: x is not None,
                                                    [re.match(r'checkpoint_(\d+)\.pth\.tar', pname) for pname in
                                                     os.listdir(modelpath)])])
        checkpoint = torch.load(os.path.join(modelpath, 'checkpoint_%s.pth.tar' % max_epoch))
        modelbest = torch.load(os.path.join(modelpath, 'model_best.pth.tar'))
        print('Loaded checkpoint %s!!!' % max_epoch)
    except:
        checkpoint = None
        modelbest = None
        print('No existing checkpoints!!!')
    return checkpoint, modelbest


if __name__ == '__main__':
    main()

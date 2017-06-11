import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNRNNModel(nn.Module):
    def __init__(self, titleninput, bodyninput,
                 nhidden, nlayers, classify_hidden,
                 nclass, title_len, body_len,
                 rnntype, dropout, conditional, bidirectional, on_cuda=False):
        super(CNNRNNModel, self).__init__()
        self.titleninput = titleninput
        self.bodyninput = bodyninput
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.classify_hidden = classify_hidden
        self.title_len = title_len
        self.body_len = body_len
        self.conditional = conditional
        self.rnntype = rnntype
        if bidirectional:
            self.chn = 2
        else:
            self.chn = 1
        self.on_cuda = on_cuda

        assert rnntype in ['LSTM', 'GRU']

        self.drop = nn.Dropout(dropout)
        self.title_rnn = getattr(nn, rnntype)(titleninput, nhidden, nlayers,
                                              batch_first=True, bidirectional=bidirectional)
        self.body_rnn = getattr(nn, rnntype)(bodyninput, nhidden, nlayers,
                                             batch_first=True, bidirectional=bidirectional)
        self.decoder_title = nn.Linear(nhidden * self.chn + titleninput, classify_hidden)
        self.decoder_body = nn.Linear(nhidden * self.chn + bodyninput, classify_hidden)
        self.tanh = nn.Tanh()
        self.maxpool_title = nn.MaxPool1d(title_len)
        self.maxpool_body = nn.MaxPool1d(body_len)
        self.classifier = nn.Linear(classify_hidden * 2, nclass)
        self.softmax = nn.Softmax()

    def _init_hidden(self, bsz):
        if self.rnntype == 'LSTM':
            if self.on_cuda:
                h0 = Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden)).cuda()
                c0 = Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden)).cuda()
            else:
                h0 = Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden))
                c0 = Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden))
            return (h0, c0)
        else:
            if self.on_cuda:
                return Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden)).cuda()
            else:
                return Variable(torch.zeros(self.nlayers * self.chn, bsz, self.nhidden))

    def forward(self, x):
        title = x[:, :self.title_len, :]
        body = x[:, self.title_len:, :]

        title_hidden = self._init_hidden(x.size(0))
        title_out, hid1 = self.title_rnn(title, title_hidden)

        if self.conditional:
            body_hidden = hid1
        else:
            body_hidden = self._init_hidden(x.size(0))
        body_out, _ = self.body_rnn(body, body_hidden)

        title_conv = torch.cat((title_out, title), dim=2).view(-1, self.nhidden * self.chn + self.titleninput)
        body_conv = torch.cat((body_out, body), dim=2).view(-1, self.nhidden * self.chn + self.bodyninput)
        title_conv = self.tanh(self.decoder_title(title_conv)).view(-1, self.title_len, self.classify_hidden)
        body_conv = self.tanh(self.decoder_body(body_conv)).view(-1, self.body_len, self.classify_hidden)
        title_conv = self.maxpool_title(torch.transpose(title_conv, 1, 2)).view(-1, self.classify_hidden)
        body_conv = self.maxpool_body(torch.transpose(body_conv, 1, 2)).view(-1, self.classify_hidden)
        tbconv = torch.cat((title_conv, body_conv), 1)

        return self.softmax(self.classifier(self.drop(tbconv)))


if __name__ == '__main__':
    titleinput = 10
    bodyinput = 5
    nhidden = 100
    nlayers = 20
    classify_hidden = 200
    nclass = 4
    title_len = 20
    body_len = 100
    bsz = 5
    wemb = 50
    input = Variable(torch.rand(bsz, title_len + body_len, wemb))

    bm = CNNRNNModel(wemb, wemb, nhidden, nlayers, classify_hidden, nclass, title_len, body_len, 'GRU', 0.1, True, True)

    print(bm.forward(input))
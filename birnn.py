import torch
import torch.nn as nn
from torch.autograd import Variable


class BiRNNModel(nn.Module):
    def __init__(self, titleninput, bodyninput,
                 nhidden, nlayers, classify_hidden,
                 nclass, title_len, body_len,
                 rnntype, dropout, conditional, bidirectional, on_cuda=False):
        super(BiRNNModel, self).__init__()
        self.nhidden = nhidden
        self.nlayers = nlayers
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
        self.decoder = nn.Linear((nhidden + nhidden) * self.chn, classify_hidden)
        self.classifier = nn.Linear(classify_hidden, nclass)
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

        output = torch.cat((title_out[:, -1, :], body_out[:, -1, :]), 1)
        class_hidden = self.drop(self.decoder(output))
        return self.softmax(self.classifier(class_hidden))

    def format_data(self, xin, yin):
        xformateed = Variable(torch.FloatTensor(xin), requires_grad=True)
        yformateed = Variable(torch.LongTensor(yin).view(-1), requires_grad=False)
        return xformateed, yformateed


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
    input = Variable(torch.rand(title_len + body_len, bsz, wemb)).cuda()

    bm = BiRNNModel(wemb, wemb, nhidden, nlayers, classify_hidden, nclass, title_len, body_len, 'LSTM')

    print(bm.forward(input))

import torch
import torch.nn as nn
from torch.autograd import Variable


class AttentionOutRNNUnit(nn.Module):
    def __init__(self, nhidden, nencoderinput, on_cuda):
        super(AttentionOutRNNUnit, self).__init__()
        self.nhidden = nhidden
        self.nencoderinput = nencoderinput
        self.on_cuda = on_cuda

        self.bilinear = nn.Bilinear(nencoderinput, nhidden, 1)
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoderhidden):
        scores = self._score(hidden, encoderhidden)
        encoderlen = encoderhidden.size(1)
        context = torch.bmm(torch.transpose(encoderhidden, 1, 2), scores.view(-1, encoderlen, 1)).view(-1, self.nencoderinput) # B x D
        return context


    def _score(self, hidden, encoderhidden):
        # hidden: B x D
        # encoderhidden: B x L x D
        bsz = encoderhidden.size(0)
        encoderlen = encoderhidden.size(1)
        scores = Variable(torch.zeros(bsz, encoderlen, 1))
        if self.on_cuda:
            scores = scores.cuda()
        for t in range(encoderlen):
            scores[:, t, :] = self.bilinear(encoderhidden[:, t, :], hidden)
        scores = scores.view(-1, encoderlen)
        scores = self.softmax(scores)  # B x L
        return scores


class GlobalNMTAttentionRNNModel(nn.Module):
    '''
    https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self, titleninput, bodyninput,
                 nhidden, nlayers, classify_hidden,
                 nclass, title_len, body_len,
                 rnntype, dropout, conditional, bidirectional, on_cuda=False):
        super(GlobalNMTAttentionRNNModel, self).__init__()
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
        self.attention = AttentionOutRNNUnit(nhidden * self.chn, nhidden * self.chn, on_cuda)
        self.linear_attention = nn.Linear((nhidden + nhidden) * self.chn, nhidden * self.chn)
        self.tanh = nn.Tanh()
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
        title_out, title_hidden = self.title_rnn(title, title_hidden)

        if self.conditional:
            body_hidden = title_hidden
        else:
            body_hidden = self._init_hidden(x.size(0))
        body_out, _ = self.body_rnn(body, body_hidden)

        body_out_context = self.attention(body_out[:, -1, :], title_out)
        body_out = self.tanh(self.linear_attention(torch.cat((body_out_context, body_out[:, -1, :]), dim=1)))

        output = torch.cat((title_out[:, -1, :], body_out), 1)
        class_hidden = self.drop(self.decoder(output))
        return self.softmax(self.classifier(class_hidden))

    def format_data(self, xin, yin):
        xformateed = Variable(torch.FloatTensor(xin), requires_grad=True)
        yformateed = Variable(torch.LongTensor(yin).view(-1), requires_grad=False)
        return xformateed, yformateed


class QAGlobalAttentionRNNModel(nn.Module):
    def __init__(self, titleninput, bodyninput,
                 nhidden, nlayers, classify_hidden,
                 nclass, title_len, body_len,
                 rnntype, dropout, conditional, bidirectional, on_cuda=False):
        super(QAGlobalAttentionRNNModel, self).__init__()
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
        self.attention = AttentionOutRNNUnit(nhidden * self.chn, nhidden * self.chn, on_cuda)
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
        title_out, title_hidden = self.title_rnn(title, title_hidden)

        if self.conditional:
            body_hidden = title_hidden
        else:
            body_hidden = self._init_hidden(x.size(0))
        body_out, _ = self.body_rnn(body, body_hidden)

        body_out = self.attention(title_out[:, -1, :], body_out)

        output = torch.cat((title_out[:, -1, :], body_out), 1)
        class_hidden = self.drop(self.decoder(output))
        return self.softmax(self.classifier(class_hidden))

    def format_data(self, xin, yin):
        xformateed = Variable(torch.FloatTensor(xin), requires_grad=True)
        yformateed = Variable(torch.LongTensor(yin).view(-1), requires_grad=False)
        return xformateed, yformateed


if __name__ == '__main__':
    attm = QAGlobalAttentionRNNModel(
            titleninput=10,
            bodyninput=10,
            nhidden=10,
            nlayers=2,
            classify_hidden=4 * 10,
            nclass=4,
            title_len=40,
            body_len=100,
            rnntype='GRU',
            dropout=0.1,
            conditional=True,
            bidirectional=True,
            on_cuda=False
    )
    # x = Variable(torch.rand(100, 140, 10), requires_grad=True)
    # print(attm(x))
    import numpy as np
    x = np.zeros((10,100,30))
    y = np.vstack(tuple([0 for _ in range(100)]))
    print(attm.format_data(x, y))

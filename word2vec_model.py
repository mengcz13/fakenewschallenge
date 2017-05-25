# -*- coding: utf-8 -*-
# Collect training data and vectorize it with GoogleNews-vectors-negatice300 model(pretrained)

import os
import re
import numpy as np
from csv import DictReader
from sklearn import feature_extraction
from gensim.models.keyedvectors import KeyedVectors
import torch
from torch.autograd import Variable


class Data(object):
    def __init__(self, datapath='./data'):
        word2vec_bin_path = os.path.join(datapath, 'GoogleNews-vectors-negative300.bin')
        train_bodies_path = os.path.join(datapath, 'train_bodies.csv')
        train_stances_path = os.path.join(datapath, 'train_stances.csv')

        with open(train_bodies_path, 'r', encoding='utf-8') as bodiesf:
            with open(train_stances_path, 'r', encoding='utf-8') as stancesf:
                print('Loading bodies and stances...')
                bodies = DictReader(bodiesf)
                stances = DictReader(stancesf)
                id2body_map = {}
                for body in bodies:
                    id2body_map[int(body['Body ID'])] = body['articleBody']

                # self.article_list = listof(headline, body, stance)
                print('Preparing article list...')
                self.article_list = []
                for stance in stances:
                    self.article_list.append((self._preprocess_text(stance['Headline']),
                                              self._preprocess_text(id2body_map[int(stance['Body ID'])]),
                                              stance['Stance']))
                print(self.article_list[0])

        self.int2stancemap = list(set([a[2] for a in self.article_list]))
        self.stance2intmap = {}
        for t in range(len(self.int2stancemap)):
            self.stance2intmap[self.int2stancemap[t]] = t

        print('Generating statistics...')
        self.maxtitlelen = max((len(a[0]) for a in self.article_list))
        self.maxbodylen = max((len(a[1]) for a in self.article_list))
        self.stancelen = len(self.int2stancemap)
        print('Max length of title: ', self.maxtitlelen)
        print('Max length of body: ', self.maxbodylen)
        print('Kinds of stance: ', self.stancelen)
        print('Stances: ', self.stance2intmap)

        print('Loading GoogleNews-vectors-negative300.bin...')
        self._model = KeyedVectors.load_word2vec_format(word2vec_bin_path, binary=True)
        self.vecdim = self._modeldim = self._model.vector_size
        print('Dim of word vector: ', self.vecdim)

        self.dataiter = self.data()

    def data(self):
        '''
        Get the iterator of all vectorized articles.
        :return: a new iterator of vectorized articles 
        '''
        return map(lambda article: (
            self._get_vec_from_words(article[0], fillzerosto=self.maxtitlelen),
            self._get_vec_from_words(article[1], fillzerosto=self.maxbodylen), self.stance2intmap[article[2]]),
                   self.article_list)

    def get_article_list(self):
        return self.article_list

    def get_batch(self, batch_size):
        binput = []
        boutput = []
        for tx in range(batch_size):
            try:
                tdata = next(self.dataiter)
            except StopIteration:
                self.dataiter = self.data()
                # tdata = next(self.dataiter)
                break
            binput.append(np.vstack((tdata[0], tdata[1])).astype(np.float64))
            boutput.append(tdata[2])
        inputshape = binput[0].shape
        binput = Variable(torch.FloatTensor(binput).view(-1, 1, inputshape[0], inputshape[1]), requires_grad=True)
        boutput = Variable(torch.LongTensor(boutput).view(-1), requires_grad=False)
        return binput, boutput

    def _clean(self, s):
        return map(lambda x: x.lower(), re.findall(r'[a-zA-Z]+', s, flags=re.UNICODE))

    def _remove_stopwords(self, l):
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS and len(w) > 1]

    def _preprocess_text(self, s):
        return self._remove_stopwords(self._clean(s))

    def _get_vec_from_words(self, s, fillzerosto=None):
        '''
        Transform string s to numpy 2d array. Fill zeros if size is smaller.
        :param s: string
        :param fillzerosto: if given, fill zeros to satisfy the given length
        :return: numpy 2d array
        '''
        vecs = None
        for word in s:
            vec = None
            try:
                vec = self._model.word_vec(word)
            except KeyError:
                vec = np.zeros(self._modeldim)
            if vecs is None:
                vecs = vec
            else:
                vecs = np.vstack((vecs, vec))
        if fillzerosto:
            tofill = fillzerosto - vecs.shape[0]
            for t in range(tofill):
                vecs = np.vstack((vecs, np.zeros(vecs.shape[1])))
        return vecs


if __name__ == '__main__':
    tdata = Data()
    for t in range(3):
        traindata = tdata.data()
        for t2 in range(1):
            print(next(traindata))

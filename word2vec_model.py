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
    def __init__(self, datapath, word2vecmodelfile, train_ratio, remove_stopwords, title_trunc, body_trunc):
        word2vec_bin_path = os.path.join(datapath, word2vecmodelfile)
        train_bodies_path = os.path.join(datapath, 'train_bodies.csv')
        train_stances_path = os.path.join(datapath, 'train_stances.csv')
        self.remove_stopwords = remove_stopwords

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
                    self.article_list.append((self._preprocess_text(stance['Headline'], trunc=title_trunc),
                                              self._preprocess_text(id2body_map[int(stance['Body ID'])],
                                                                    trunc=body_trunc),
                                              stance['Stance']))
                print(self.article_list[0])

        self.int2stancemap = list(set([a[2] for a in self.article_list]))
        self.stance2intmap = {}
        for t in range(len(self.int2stancemap)):
            self.stance2intmap[self.int2stancemap[t]] = t

        print('Generating statistics...')
        self.maxtitlelen = max((len(a[0]) for a in self.article_list))
        print('Max length of title:', self.maxtitlelen)
        self.maxtitlelen = max((self.maxtitlelen, title_trunc))
        print('Truncated title length:', self.maxtitlelen)
        self.maxbodylen = max((len(a[1]) for a in self.article_list))
        print('Max length of body:', self.maxbodylen)
        self.maxbodylen = max((self.maxbodylen, body_trunc))
        print('Truncated body length:', self.maxbodylen)
        self.stancelen = len(self.int2stancemap)
        print('Kinds of stance:', self.stancelen)
        print('Stances:', self.stance2intmap)

        print('Loading %s' % word2vecmodelfile)
        self._model = KeyedVectors.load_word2vec_format(word2vec_bin_path, binary=True)
        self.vecdim = self._modeldim = self._model.vector_size
        print('Dim of word vector:', self.vecdim)

        self.trainsize = round(train_ratio * len(self.article_list))
        self.validatesize = len(self.article_list) - self.trainsize
        trainsize = self.trainsize
        self.train_article_list = self.article_list[:trainsize]
        self.validate_article_list = self.article_list[trainsize:]

        self.trainvecs = list(self._data(self.train_article_list))
        self.validatevecs = list(self._data(self.validate_article_list))

        self.train_dataiter = iter(self.trainvecs)
        self.validate_dataiter = iter(self.validatevecs)

    def train_data(self):
        return iter(self.trainvecs)

    def validate_data(self):
        return iter(self.validatevecs)

    def train_get_batch(self, batch_size):
        binput = []
        boutput = []
        for tx in range(batch_size):
            try:
                tdata = next(self.train_dataiter)
            except StopIteration:
                self.train_dataiter = self.train_data()
                break
            binput.append(tdata[0].astype(np.float64))
            boutput.append(tdata[1])
        inputshape = binput[0].shape
        binput = torch.FloatTensor(binput).view(-1, 1, inputshape[0], inputshape[1])
        boutput = torch.LongTensor(boutput).view(-1)
        return binput, boutput

    def validate_get_batch(self, batch_size):
        binput = []
        boutput = []
        for tx in range(batch_size):
            try:
                tdata = next(self.validate_dataiter)
            except StopIteration:
                self.validate_dataiter = self.validate_data()
                break
            binput.append(tdata[0].astype(np.float64))
            boutput.append(tdata[1])
        inputshape = binput[0].shape
        binput = torch.FloatTensor(binput).view(-1, 1, inputshape[0], inputshape[1])
        boutput = torch.LongTensor(boutput).view(-1)
        return binput, boutput

    def _data(self, alist):
        '''
        Get the iterator of all vectorized articles.
        :return: a new iterator of vectorized articles 
        '''
        return map(lambda article: (
            self._get_vec_from_words(article[0], self.maxtitlelen, article[1], self.maxbodylen),
            self.stance2intmap[article[2]]),
                   alist)

    def get_article_list(self):
        return self.article_list

    def _clean(self, s):
        return map(lambda x: x.lower(), re.findall(r'[a-zA-Z]+', s, flags=re.UNICODE))

    def _remove_stopwords(self, l):
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS and len(w) > 1]

    def _preprocess_text(self, s, trunc):
        if self.remove_stopwords:
            res = self._remove_stopwords(self._clean(s))
        else:
            res = list(self._clean(s))
        return res[:trunc]

    def _get_vec_from_words(self, s0, fillzerosto0, s1, fillzerosto1):
        lens0 = len(s0)
        lens1 = len(s1)
        vecs = [np.zeros(self._modeldim) for t in range(fillzerosto0 + fillzerosto1)]
        for t in range(lens0):
            try:
                vecs[t] = self._model.word_vec(s0[t])
            except KeyError:
                pass
        for t in range(lens1):
            try:
                vecs[t + fillzerosto0] = self._model.word_vec(s1[t])
            except KeyError:
                pass
        return np.vstack(tuple(vecs))


if __name__ == '__main__':
    pass

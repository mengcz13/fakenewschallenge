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
import random
import collections


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
                random.shuffle(self.article_list)
                print(self.article_list[0])

        self.int2stancemap = sorted(list(set([a[2] for a in self.article_list])))
        self.stance2intmap = {}
        for t in range(len(self.int2stancemap)):
            self.stance2intmap[self.int2stancemap[t]] = t

        print('Generating statistics...')
        self.maxtitlelen = max((len(a[0]) for a in self.article_list))
        print('Max length of title:', self.maxtitlelen)
        self.maxtitlelen = min((self.maxtitlelen, title_trunc))
        print('Truncated title length:', self.maxtitlelen)
        self.maxbodylen = max((len(a[1]) for a in self.article_list))
        print('Max length of body:', self.maxbodylen)
        self.maxbodylen = min((self.maxbodylen, body_trunc))
        print('Truncated body length:', self.maxbodylen)
        self.stancelen = len(self.int2stancemap)
        print('Kinds of stance:', self.stancelen)
        print('Stances:', self.stance2intmap)

        self.trainsize = round(train_ratio * len(self.article_list))
        self.validatesize = len(self.article_list) - self.trainsize
        trainsize = self.trainsize
        self.train_article_list = self.article_list[:trainsize]
        self.validate_article_list = self.article_list[trainsize:]

        # random sample on train article list
        train_stance_col = collections.Counter([a[2] for a in self.train_article_list])
        mc = train_stance_col.most_common(1)[0]
        for st, cn in train_stance_col.items():
            if st != mc[0]:
                assert cn <= mc[1]
                pop = [a for a in self.train_article_list if a[2] == st]
                self.train_article_list.extend(random.choices(pop, k=mc[1] - cn))
                self.trainsize += (mc[1] - cn)
        random.shuffle(self.train_article_list)
        print(collections.Counter([a[2] for a in self.train_article_list]))

        print('Loading %s' % word2vecmodelfile)
        isbinary = word2vecmodelfile.split('.')[-1] == 'bin'
        self._model = KeyedVectors.load_word2vec_format(word2vec_bin_path, binary=isbinary)
        self.vecdim = self._modeldim = self._model.vector_size
        print('Dim of word vector:', self.vecdim)

        self.trainvecs, self.trainres = self._data(self.train_article_list)
        self.validatevecs, self.validateres = self._data(self.validate_article_list)

    def train_batch(self, batch_size):
        return self._batch(self.trainvecs, self.trainres, self.trainsize, batch_size)

    def validate_batch(self, batch_size):
        return self._batch(self.validatevecs, self.validateres, self.validatesize, batch_size)

    def _batch(self, vecs, res, vecslen, batch_size):
        t = 0
        while t < vecslen:
            if t + batch_size < vecslen:
                absize = batch_size
            else:
                absize = vecslen - t
            yield vecs[t:t + batch_size], res[t:t + batch_size], t // batch_size, absize
            t += batch_size

    def _data(self, alist):
        return list(map(lambda article:
                        self._get_vec_from_words(article[0], self.maxtitlelen, article[1], self.maxbodylen), alist)), \
               list(map(lambda article:
                        self.stance2intmap[article[2]], alist))

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
        return np.vstack(tuple(vecs)).astype(np.float64)


if __name__ == '__main__':
    pass

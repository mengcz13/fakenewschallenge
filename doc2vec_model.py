import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from csv import DictReader
from argparse import ArgumentParser


class OriginDoc(object):
    def __init__(self, datapath):
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
                    self.article_list.append((stance['Headline'],
                                              id2body_map[int(stance['Body ID'])],
                                              stance['Stance']))
                print(self.article_list[0])

    def read_corups(self):
        self.corups = []
        for i, line in enumerate(self.article_list):
            self.corups.append(TaggedDocument(gensim.utils.simple_preprocess(line[0]), [i * 2]))
            self.corups.append(TaggedDocument(gensim.utils.simple_preprocess(line[1]), [i * 2 + 1]))
        return self.corups


class Doc2VecModel(object):
    def __init__(self, size, min_count, iter, datapath, dotrain):
        modelpath = os.path.join(datapath, 'doc2vec_model.model')
        if not dotrain and os.path.exists(modelpath):
            self.model = Doc2Vec.load(modelpath)
        else:
            self.model = self._train(size, min_count, iter, datapath, modelpath)

    def _train(self, size, min_count, iter, datapath, modelpath):
        originDoc = OriginDoc(datapath)
        corups = originDoc.read_corups()
        model = Doc2Vec(size=size, min_count=min_count, iter=iter)
        print('Building vocab...')
        model.build_vocab(corups)
        print('Training...')
        model.train(corups, total_examples=model.corpus_count, epochs=model.iter)
        print('Finished! Save model to %s...' % (modelpath))
        model.save(modelpath)
        print(model.docvecs[0], model.docvecs[1])
        return model

    def docvec(self):
        return self.model.docvecs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data',
                        help='path to data folder')
    parser.add_argument('--size', type=int, default=300,
                        help='size of doc vec')
    parser.add_argument('--min_count', type=int, default=2,
                        help='min_count')
    parser.add_argument('--iter', type=int, default=55,
                        help='iter times')
    parser.add_argument('--dotrain', action='store_true',
                        help='flag for do training')
    options = parser.parse_args()

    doc2vecmodel = Doc2VecModel(size=options.size, min_count=options.min_count, iter=options.iter,
                                datapath=options.datapath, dotrain=options.dotrain)
    docvecs = doc2vecmodel.docvec()
    print(docvecs[0], docvecs[1], docvecs[2], docvecs[3])

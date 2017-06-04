from argparse import ArgumentParser
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('glovefile', type=str)
    args.add_argument('word2vecfile', type=str)
    options = args.parse_args()
    glove2word2vec(options.glovefile, options.word2vecfile)
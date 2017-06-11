from multiprocessing import Pool
import os, time, random
import sys
import subprocess

PY = '/home/ubuntu/.pyenv/versions/miniconda3-latest/bin/python main.py '


def long_time_task(command, outfilename):
    print('Run task %s (%s)..., print to %s' % (command, os.getpid(), outfilename))
    start = time.time()
    with open(outfilename, 'w+') as f:
        subprocess.call((PY + command).split(' '), stdout=f)
    end = time.time()
    print('Task %s runs %0.2f seconds, print to %s' % (command, (end - start), outfilename))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(2)

    comms = []

    # Dimension comparision
    # dims = [50, 100, 200]
    # comms.extend([
    #     '--cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_%sd --word2vecmodelfile glove.twitter.27B.%sd.txt --fix_embedding --epoch 25' % (
    #     t, t) for t in dims
    # ])
    # comms.extend([
    #     '--cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_selfemb_50d --selfemb 50 --epoch 25',
    #     '--cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_halfselfemb_50d --word2vecmodelfile glove.twitter.27B.50d.txt --epoch 25',
    # ])

    # models
    models = ['cnn', 'rnn_gru_cond_bi', 'rnn_gru_nocond_bi', 'globalnmt_attention_gru_cond_bi',
              'qaglobal_attention_gru_cond_bi', 'cnn_rnn']
    comms.extend([
        '--cuda %s savedmodels/%s --word2vecmodelfile glove.twitter.27B.50d.txt --fix_embedding --epoch 200 --resume' % (
        t, t)
        for t in
        models
    ])

    # Body length
    body_lens = [100, 200, 300]
    comms.extend([
        '--cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_%sbody --word2vecmodelfile glove.twitter.27B.50d.txt --fix_embedding --body_trunc %s --epoch 25' % (
            t, t) for t in body_lens
    ])

    # LSTM and GRU
    comms.extend([
        '--cuda rnn_lstm_cond_bi savedmodels/rnn_lstm_cond_bi_lstm --word2vecmodelfile glove.twitter.27B.50d.txt --fix_embedding --epoch 25',
        '--cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_gru --word2vecmodelfile glove.twitter.27B.50d.txt --fix_embedding --epoch 25',
    ])

    for t, comm in enumerate(comms):
        p.apply_async(long_time_task, args=(comm, 'task_%s.log' % t))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

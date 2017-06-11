实验计划:
比较: 维数/正文截取长度/lstm和gru/不同模型

```
python main.py --cuda rnn_gru_cond_bi savedmodels/rnn_gru_cond_bi_50d --word2vecmodelfile glove.twitter.27B.50d.txt --fix_embedding
```

维数 on birnn
50
100
200

正文长度 on birnn
100
200
300


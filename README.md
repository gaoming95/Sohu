# Sohu 重要实体抽取

情感为实体的0.6左右，但是是队友的代码，不开源

菜鸟开源 综合成绩0.44336

# 1 SouhuNER

（1）自己写的一个传统的Bi-LSTM-CRF NER模型，可能和主流代码都差不多

（2）基于字，random Embedding 和 BERT的Pre_train

（3）实体结果应该在0.43-0.44之间（没有优化）

# 2 SohuSummary

（1）偷懒，不想做管道模型，想做端到端的关键实体抽取，于是做了一个Seq2Seq

（2）基于字，random Embedding 和 BERT的Pre_train

（3）实体结果应该在0.46-0.47之间（没有优化）

# 3 Bert_Souhu

（1）没办法，BERT太强大，没有用Google的BERT源码，而是第三方的一个包，感觉只要跑完就有0.54的分数

（2）基于BERT的fine_tuning

（3）实体结果应该在0.54-0.56之间

# 想法：

（1）没有卡就不要打比赛了，靠一张1080苟活到现在

（2）还是要看一下讨论区，闭门造车死得惨

（3）一开始偷懒了，想做一个端对端的模型。所以没考虑实体的重要性如何去做，之后实在太赶，就没做了，取前三效果也不错。

（4）BERT大法好（当然有机会想看以下前排大佬的代码）

# 注意：

有的地方代码看似来很愚蠢，代码没有润色就上传了，可能把许多文件合并在一起。所以读写文件读了很多次，我是面向脚本编程的！

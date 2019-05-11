# 重载模型,对一句话提取实体
# 加载数据
import tensorflow as tf
import os

from model import SouHuNER
from util import InputHelper

tf.flags.DEFINE_integer('clip_grad', 5, 'clip_grad')
tf.flags.DEFINE_integer('embedding_dim', 300, 'word embedding dim')
tf.flags.DEFINE_integer('hidden_dim', 300, 'hidden_dim')
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

line_helper = InputHelper('data', None, 1)
with open('./data/train_key', 'r', encoding='utf-8') as g:
    test_data = g.readlines()
# string = '玻璃钢脱硫除尘器净化塔砖厂隧道窑脱硫塔洗涤塔锅炉脱硫厂家直销玻璃钢脱硫除尘器净化塔砖厂隧道窑脱硫塔洗涤塔锅炉脱硫厂家直销玻璃钢脱硫塔是对工业废气进行脱'
# word_id = [line_helper.sentence2id(string)]
# print(word_id)
num_tags = line_helper.num_tags
id2tag = {line_helper.tag2label[key]: key for key in line_helper.tag2label}
# embedding = line_helper.embedding
vocab_size = line_helper.vocab_size
model = SouHuNER(vocab_size, FLAGS.embedding_dim, FLAGS.hidden_dim, num_tags, FLAGS.clip_grad)
saver = tf.train.Saver()
# 重载模型
with open('./test', 'w', encoding='utf-8') as g_key:
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model', '1556081029', 'checkpoint')))

        for line in test_data:
            line_keys = []
            for sent in line.strip().split('\t'):
                sent_id = [line_helper.sentence2id(sent)]
                feed, word_length = model.create_feed_dict(sent_id)
                logits, transition_params = sess.run([model.logits, model.transition_params], feed_dict=feed)
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logits[0][:word_length[0]], transition_params)
                label = line_helper.iob_iobes([id2tag[v] for v in viterbi_seq])
                result = line_helper.result_to_json(sent, label)
                line_keys += [entity['word'] for entity in result['entities']]
            g_key.write(','.join(line_keys) + '\n')

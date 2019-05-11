import tensorflow as tf
import os
from model import Model
from util import InputHelper

tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('train_file', 'train', 'train_file')
tf.flags.DEFINE_string('test_file', 'test', 'test_file')
tf.flags.DEFINE_string('model_dir', 'model', 'model directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model save directory')
tf.flags.DEFINE_string('ckpt_dir', 'checkpoint', 'checkpoint save directory')
tf.flags.DEFINE_integer('embedding_dim', 300, 'word embedding dim')
tf.flags.DEFINE_integer('hidden_dim', 128, 'hidden_dim default:')
tf.flags.DEFINE_integer('num_layers', 1, 'num_layers default:2')
tf.flags.DEFINE_integer('beam_width', 10, 'beam_width')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate (default : 0.001)')
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('num_epochs', 10, 'num_epochs (default : 32)')
tf.flags.DEFINE_integer('clip_grad', 5, 'clip_grad')
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

# 文件测试
play_helper = InputHelper(FLAGS.data_dir, FLAGS.test_file, FLAGS.batch_size, toy=True)
vocab_size = play_helper.vocab_size
word2id = play_helper.word2id
id2word = dict(zip(word2id.values(), word2id.keys()))
model = Model(FLAGS.hidden_dim, FLAGS.num_layers, vocab_size, FLAGS.embedding_dim, FLAGS.beam_width, forward_only=True)
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(os.path.join('model', '1556093025', 'checkpoint'))
    saver.restore(sess, ckpt)
    batches = play_helper.batch_yield(shuffle=False)
    for step, (sents, keys) in enumerate(batches):
        feed = model.create_feed_dict(sents, keys, word2id, None, 1)
        prediction = sess.run(model.prediction, feed_dict=feed)
        prediction_output = [[id2word[y] for y in x] for x in prediction[:, 0, :]]
        with open("result.txt", "a",encoding='utf-8') as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    # if word not in summary:
                    summary.append(word)
                print(" ".join(summary), file=f)
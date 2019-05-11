import tensorflow as tf
import os
import time
from model import Model
from util import InputHelper
import numpy as np

tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('train_file', 'train_1', 'train_file')
tf.flags.DEFINE_string('test_file', 'valid_1', 'test_file')
tf.flags.DEFINE_string('model_dir', 'model', 'model directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model save directory')
tf.flags.DEFINE_string('ckpt_dir', 'checkpoint', 'checkpoint save directory')
tf.flags.DEFINE_integer('embedding_dim', 300, 'word embedding dim')
tf.flags.DEFINE_integer('hidden_dim', 128, 'hidden_dim default: 150')
tf.flags.DEFINE_integer('num_layers', 1, 'num_layers default:2')
tf.flags.DEFINE_integer('beam_width', 10, 'beam_width')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate (default : 0.001)')
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('num_epochs', 10, 'num_epochs (default : 32)')
tf.flags.DEFINE_integer('clip_grad', 5, 'clip_grad')
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

timestamp = str(int(time.time()))
log_path = os.path.join(FLAGS.model_dir, timestamp, FLAGS.log_dir)
if not os.path.exists(log_path):
    os.makedirs(log_path)
save_path = os.path.join(FLAGS.model_dir, timestamp, FLAGS.save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)
ckpt_path = os.path.join(FLAGS.model_dir, timestamp, FLAGS.ckpt_dir)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

train_data_loader = InputHelper(FLAGS.data_dir, FLAGS.train_file, FLAGS.batch_size, toy=True)
test_data_loader = InputHelper(FLAGS.data_dir, FLAGS.test_file, FLAGS.batch_size, toy=True)
vocab_size = train_data_loader.vocab_size

train_model = Model(FLAGS.hidden_dim, FLAGS.num_layers, vocab_size, FLAGS.embedding_dim, FLAGS.beam_width,
                    forward_only=False)

# log information
logger = train_data_loader.get_logger(os.path.join(log_path, 'log.txt'))

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter(save_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    num_batches = (train_data_loader.data_size + FLAGS.batch_size - 1) // FLAGS.batch_size

    for e in range(FLAGS.num_epochs):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        train_batches = train_data_loader.batch_yield(shuffle=True)
        for step, (sents, keys) in enumerate(train_batches):
            feed = train_model.create_feed_dict(sents, keys, train_data_loader.word2id, FLAGS.lr, FLAGS.dropout)
            train_loss, step_num, _ = sess.run(
                [train_model.loss, train_model.global_step, train_model.train_op],
                feed_dict=feed)
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
            file_writer.add_summary(train_summary, step_num)
            log_info = '{} epoch {}/{}, step {}/{}, loss:{:.6}, global_step: {}'.format(start_time, e + 1,
                                                                                        FLAGS.num_epochs, step + 1,
                                                                                        num_batches,
                                                                                        train_loss, step_num)
            print(log_info)
            if step + 1 == 1 or step + 1 == num_batches:
                logger.info(log_info)
                saver.save(sess, os.path.join(ckpt_path, 'ckpt'), global_step=step_num)
        # ==================测试误差==================
        logger.info('==================测试误差==================')
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        test_batches = test_data_loader.batch_yield(shuffle=False)
        test_res = []
        for step, (sents, keys) in enumerate(test_batches):
            feed = train_model.create_feed_dict(sents, keys, test_data_loader.word2id, 1, 1)
            test_loss = sess.run([train_model.loss], feed_dict=feed)
            test_res.append(test_loss)
        test_res_loss = np.average(test_res)
        test_summary = tf.Summary(value=[tf.Summary.Value(tag='valid_loss', simple_value=test_res_loss)])
        file_writer.add_summary(test_summary, e)
        log_info = '{} valid epoch:{} loss:{:.6}'.format(start_time, e, test_res_loss)
        print(log_info)
        logger.info(log_info)

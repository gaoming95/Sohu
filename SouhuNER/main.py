import tensorflow as tf
import os
import time
from conlleval import evaluate
from model import SouHuNER

from util import InputHelper

tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('train_file', 'train', 'train_file')
tf.flags.DEFINE_string('test_file', 'valid', 'test_file')
tf.flags.DEFINE_string('model_dir', 'model', 'model directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model save directory')
tf.flags.DEFINE_string('ckpt_dir', 'checkpoint', 'checkpoint save directory')
tf.flags.DEFINE_integer('embedding_dim', 300, 'word embedding dim')
tf.flags.DEFINE_integer('hidden_dim', 300, 'hidden_dim')
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

train_data_loader = InputHelper(FLAGS.data_dir, FLAGS.train_file, FLAGS.batch_size)
# 可以拆分数据集进行验证测试
# test_data_loader = InputHelper(FLAGS.data_dir, FLAGS.test_file, FLAGS.batch_size)
vocab_size = train_data_loader.vocab_size
num_tags = train_data_loader.num_tags
# embedding = train_data_loader.embedding

model = SouHuNER(vocab_size, FLAGS.embedding_dim, FLAGS.hidden_dim, num_tags, FLAGS.clip_grad)

tf.summary.scalar('train_loss', model.loss)
merged = tf.summary.merge_all()

# log information
logger = train_data_loader.get_logger(os.path.join(log_path, 'log.txt'))


def train():
    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter(save_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        num_batches = (train_data_loader.data_size + FLAGS.batch_size - 1) // FLAGS.batch_size
        for e in range(FLAGS.num_epochs):
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            train_batches = train_data_loader.batch_yield(shuffle=True)
            for step, (seqs, labels) in enumerate(train_batches):
                feed, _ = model.create_feed_dict(seqs, labels, FLAGS.lr, FLAGS.dropout)
                train_loss, summary, step_num, _ = sess.run([model.loss, merged, model.global_step, model.train_op],
                                                            feed_dict=feed)
                file_writer.add_summary(summary, step_num)
                log_info = '{} epoch {}/{}, step {}/{}, loss:{:.6}, global_step: {}'.format(start_time, e + 1,
                                                                                            FLAGS.num_epochs, step + 1,
                                                                                            num_batches,
                                                                                            train_loss, step_num)
                print(log_info)
                if step + 1 == 1 or step + 1 == num_batches:
                    logger.info(log_info)
                    saver.save(sess, os.path.join(ckpt_path, 'ckpt'), global_step=step_num)
                # if (step + 1) % 500 == 0:
                #     # 一个epoch结束进行测试
                #     logger.info("========================Test========================")
                #     test(sess, test_data_loader)


# 测试数据集
def test(sess, test_data_loader):
    test_batches = test_data_loader.batch_yield()
    labels_pred_list = []
    label_true_list = []
    for step, (seqs, labels) in enumerate(test_batches):
        feed, word_length = model.create_feed_dict(seqs, labels)
        logits, transition_params = sess.run([model.logits, model.transition_params], feed_dict=feed)
        # 维特比解码
        labels_pred_list.extend(decode(logits, transition_params, word_length))
        label_true_list.extend(labels)
    prec, rec, f1 = evaluate_f1(labels_pred_list, label_true_list)
    logger.info('prec:{} rec:{} f1:{}'.format(prec, rec, f1))


# 维特比解码
def decode(logits, transition_params, word_length):
    label_list = []
    for logit, seq_len in zip(logits, word_length):
        viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
        label_list.append(viterbi_seq)
    return label_list


# conlleval评估
def evaluate_f1(labels_pred_list, label_true_list):
    id2tag = {test_data_loader.tag2label[key]: key for key in test_data_loader.tag2label}
    res_pred = []
    for label_ in labels_pred_list:
        res_pred.extend([id2tag[l] for l in label_])
    res_true = []
    for label_ in label_true_list:
        res_true.extend([id2tag[l] for l in label_])
    prec, rec, f1 = evaluate(res_pred, res_true, verbose=True)
    return prec, rec, f1


if __name__ == '__main__':
    train()

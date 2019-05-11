import tensorflow as tf


class SouHuNER(object):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, num_tags, clip_grad):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None])

        self.dropout_pl = tf.placeholder(tf.float32, shape=[])
        self.lr_pl = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope('word_embeddings'), tf.device('/cpu:0'):
            embedding = self.weight_variables([vocab_size, embedding_dim], 'embedding')
            # _word_embedding = tf.Variable(embedding, dtype=tf.float32, trainable=True, name='_word_embedding')

            word_input_ = tf.nn.embedding_lookup(embedding, self.word_ids, name='word_input_')

        word_input = tf.nn.dropout(word_input_, self.dropout_pl)

        # BiLSTM
        with tf.variable_scope('BiLSTM'):
            cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim)
            cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=word_input,
                sequence_length=self.sequence_length,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        # logits
        with tf.variable_scope('logits'):
            w = tf.get_variable(name="W",
                                shape=[2 * hidden_dim, num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * hidden_dim])
            pred = tf.matmul(output, w) + b

            self.logits = tf.reshape(pred, [-1, s[1], num_tags])
        # loss
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                   tag_indices=self.labels,
                                                                                   sequence_lengths=self.sequence_length)
        self.loss = -tf.reduce_mean(log_likelihood)

        # train optimization
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def create_feed_dict(self, word_id, labels=None, lr=None, dropout=1):
        word_batch, word_len_batch = self.pad_sequences(word_id, pad_mark=0)
        feed_dict = {self.word_ids: word_batch, self.sequence_length: word_len_batch}
        if labels is not None:
            label_batch, _ = self.pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = label_batch
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, word_len_batch

    def weight_variables(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    def bias_variables(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def pad_sequences(self, sequences, pad_mark=0):
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list


if __name__ == '__main__':
    SouHuNER(300, 300, 256, 10, 5)

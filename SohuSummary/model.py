import tensorflow as tf


class Model(object):
    def __init__(self, rnn_size, num_layers, vocab_size, embedding_dim, beam_width, forward_only=False):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.encoder_length = tf.placeholder(tf.int32, [None])

        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_length = tf.placeholder(tf.int32, [None])

        self.decoder_target = tf.placeholder(tf.int32, [None, None])

        self.dropout_pl = tf.placeholder(tf.float32, shape=[])
        self.lr_pl = tf.placeholder(tf.float32, shape=[])
        self.global_step = tf.Variable(0, trainable=False)
        self.valid_global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(vocab_size, use_bias=False)

        # Scale and shift
        # with tf.name_scope('Scale_and_shift'):
        #     # print(tf.shape(self.encoder_inputs)[0])
        #     x_mask = tf.cast(tf.greater(tf.expand_dims(self.encoder_inputs, 2), 0), tf.float32)
        #     x = tf.cast(self.encoder_inputs, tf.int32)
        #     x = tf.one_hot(x, depth=vocab_size, axis=-1)
        #     x = tf.reduce_sum(tf.multiply(x_mask, x), 1, keep_dims=True)
        #     x = tf.cast(tf.greater(x, 0.5), tf.float32)
        #     x = tf.reshape(x, [-1, vocab_size])
        #     # kernel_shape = (1,) * (tf.shape(x)[0] - 1) + (tf.shape(x)[-1],)
        #     log_scale = tf.get_variable(name='Scale', shape=[1, vocab_size], initializer=tf.zeros_initializer(),
        #                                 dtype=tf.float32)
        #     shift = tf.get_variable(name='shift', shape=[1, vocab_size], initializer=tf.zeros_initializer(),
        #                             dtype=tf.float32)
        #     # shift = tf.Variable(tf.constant(0.1, shape=[tf.shape(x)[0],vocab_size]), name='Shift')
        #     x_prior = tf.add(tf.multiply(x, tf.exp(log_scale)), shift)

        # embedding
        with tf.name_scope('embedding'), tf.device('/cpu:0'):
            init_embeddings = tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_embedding = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs),
                                                  perm=[1, 0, 2])
            self.decoder_embedding = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input),
                                                  perm=[1, 0, 2])

        with tf.name_scope('encoder'):
            fw_cells = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)]
            bw_cells = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)]
            fw_cells = [tf.contrib.rnn.DropoutWrapper(cell) for cell in fw_cells]
            bw_cells = [tf.contrib.rnn.DropoutWrapper(cell) for cell in bw_cells]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_embedding, sequence_length=self.encoder_length, time_major=True,
                dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size * 2)
            if not forward_only:
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    rnn_size * 2, attention_states, memory_sequence_length=self.encoder_length, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=rnn_size * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=tf.shape(self.encoder_inputs)[0])
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embedding, self.decoder_length, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                # self.logits = tf.divide(tf.add(self.logits, tf.expand_dims(x_prior, 1)), 2)
                self.logits_reshape = tf.concat(
                    [self.logits,
                     tf.zeros([tf.shape(self.encoder_inputs)[0],
                               tf.reduce_max(self.decoder_length) - tf.shape(self.logits)[1], vocab_size])],
                    axis=1)

            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                                          multiplier=beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.encoder_length, multiplier=beam_width)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    rnn_size * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=rnn_size * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                        batch_size=tf.shape(self.encoder_inputs)[0] * beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.shape(self.encoder_inputs)[0]], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=tf.reduce_max(self.decoder_length),
                    scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])
        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_length, tf.reduce_max(self.decoder_length), dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(tf.shape(self.encoder_inputs)[0]))
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.lr_pl)
                self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def create_feed_dict(self, sents_id, keys_id=None, word2id=None, lr=None, dropout=1):
        batch_encoder_input, batch_encoder_input_len = self.pad_sequences(sents_id, pad_mark=0)
        feed_dict = {self.encoder_inputs: batch_encoder_input, self.encoder_length: batch_encoder_input_len}
        if keys_id is not None:
            batch_decoder_input = list(map(lambda x: [word2id["<s>"]] + list(x), keys_id))
            batch_decoder_input, batch_decoder_input_len = self.pad_sequences(batch_decoder_input, pad_mark=0)
            batch_decoder_output = list(map(lambda x: list(x) + [word2id["</s>"]], keys_id))
            batch_decoder_output, _ = self.pad_sequences(batch_decoder_output, pad_mark=0)
            feed_dict[self.decoder_input] = batch_decoder_input
            feed_dict[self.decoder_length] = batch_decoder_input_len
            feed_dict[self.decoder_target] = batch_decoder_output
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict

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
    Model(100, 1, 100, 100, 3, False)

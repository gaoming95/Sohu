import os
import random
import pickle
import logging


class InputHelper(object):
    def __init__(self, data_dir, input_file, batch_size, toy=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data = self.read_corpus(os.path.join(data_dir, input_file), toy)
        self.data_size = len(self.data)
        self.vocab_file = os.path.join(data_dir, 'vocab.pkl')
        self.word2id = self.read_dictionary(self.vocab_file)
        self.vocab_size = len(self.word2id)

    def read_corpus(self, corpus_path, toy):
        with open(corpus_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        data = []
        for line in lines:
            try:
                sent, key = line.strip().split('\t')
            except ValueError:
                sent, key = line.strip(), ' '.join(['<padding>'] * len(line.strip()))
            if toy:
                data.append(([sent.strip()[i] for i in range(0, len(sent.strip()), 2)][:1000], [key.strip()[i] for i in range(0, len(key.strip()), 2)]))
            else:
                data.append(([sent.strip()[i] for i in range(0, len(sent.strip()), 2)], [key.strip()[i] for i in range(0, len(key.strip()), 2)]))
        return data

    def sentence2id(self, sent):
        sentence_id = []
        for word in sent:
            if word not in self.word2id:
                word = '<unk>'
            sentence_id.append(self.word2id[word])
        return sentence_id

    def write_dict(self, dictfile, path):
        with open(path, 'wb') as fw:
            pickle.dump(dictfile, fw)

    def read_dictionary(self, vocab_path):
        with open(vocab_path, 'rb') as fr:
            dictionary = pickle.load(fr)
        return dictionary

    def batch_yield(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data)

        sents, keys = [], []
        for (sent_, key_) in self.data:
            sent_ = self.sentence2id(sent_)
            key_ = self.sentence2id(key_)

            if len(sents) == self.batch_size:
                yield sents, keys
                sents, keys = [], []
            sents.append(sent_)
            keys.append(key_)

        if len(sents) != 0:
            yield sents, keys

    def get_logger(self, filename):
        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        return logger


if __name__ == '__main__':

    datahelp = InputHelper('data', 'valid_1', 1, toy=True)
    for x, y in datahelp.batch_yield():
        print(x[0])
        exit(0)

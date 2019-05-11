import pickle
import os
import random
import logging
import numpy as np

class InputHelper(object):
    def __init__(self, data_dir, input_file, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        if input_file != None:
            self.data = self.read_corpus(os.path.join(data_dir, input_file))
            self.data_size = len(self.data)

        self.tag_file = os.path.join(data_dir, 'tag2label.pkl')
        self.vocab_file = os.path.join(data_dir, 'vocab.pkl')
        # self.embedding_file = os.path.join(data_dir, 'embedding.pkl')

        if not os.path.exists(self.tag_file):
            self.tags2label()
        else:
            self.tag2label = self.read_dictionary(self.tag_file)
            print(self.tag2label)
            self.num_tags = len(self.tag2label)
        if not os.path.exists(self.vocab_file):
            self.vocab_build(2)
        else:
            self.word2id = self.read_dictionary(self.vocab_file)
            self.vocab_size = len(self.word2id)
        # if not os.path.exists(self.embedding_file):
        #     raise FileNotFoundError('Embedding File not found, please use word2vec or bert to achieve word embeddings')
        # else:
        #     self.embedding = np.array(self.read_dictionary(self.embedding_file))

    def read_corpus(self, corpus_path):
        data = []
        with open(corpus_path, encoding='utf-8') as fr:
            lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                try:
                    [char, label] = line.strip().split('\t')
                except ValueError:
                    continue
                sent_.append(char)
                tag_.append(label)
            else:
                if len(sent_) != 0:
                    data.append((sent_[:1000], tag_[:1000]))
                sent_, tag_ = [], []
        return data

    def vocab_build(self, min_count):
        word2id = {}
        for sent_, tag_ in self.data:
            for word in sent_:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word and word <= '\u005a') or ('\u0061' <= word and word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
        low_freq_words = []
        for word, [word_id, word_freq] in word2id.items():
            if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

        new_id = 1
        for word in word2id.keys():
            word2id[word] = new_id
            new_id += 1
        word2id['<UNK>'] = new_id
        word2id['<PAD>'] = 0
        self.word2id = word2id
        self.vocab_size = len(self.word2id)
        print(len(word2id))
        self.write_dict(word2id, self.vocab_file)

    def sentence2id(self, sent):
        sentence_id = []
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word and word <= '\u005a') or ('\u0061' <= word and word <= '\u007a'):
                word = '<ENG>'
            if word not in self.word2id:
                word = '<UNK>'
            sentence_id.append(self.word2id[word])
        return sentence_id

    def write_dict(self, dictfile, path):
        with open(path, 'wb') as fw:
            pickle.dump(dictfile, fw)

    def read_dictionary(self, vocab_path):
        with open(vocab_path, 'rb') as fr:
            dictionary = pickle.load(fr)
        return dictionary

    def tags2label(self):
        tag2label = {'O': 0}
        tags = [t for train in self.data for t in train[1]]
        tags = set(tags)
        tags.remove('O')
        for label, tag in enumerate(sorted(list(tags), key=lambda x: str(x).split('-')[-1][0] + str(x)[0]), start=1):
            tag2label[tag] = label
        self.write_dict(tag2label, self.tag_file)
        self.tag2label = tag2label
        self.num_tags = len(tag2label)

    def batch_yield(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data)

        seqs, labels = [], []
        for (sent_, tag_) in self.data:
            sent_ = self.sentence2id(sent_)
            label_ = [self.tag2label[tag] for tag in tag_]

            if len(seqs) == self.batch_size:
                yield seqs, labels
                seqs, labels = [], []

            seqs.append(sent_)
            labels.append(label_)

        if len(seqs) != 0:
            yield seqs, labels

    def iob_iobes(self, tags):
        """
        IOB -> IOBES
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':
                if i + 1 != len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags

    def result_to_json(self, string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item

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
    data_loader = InputHelper('data', 'train', 128)
    batches = data_loader.batch_yield(shuffle=False)
    for step, (seqs, labels) in enumerate(batches):
        print(seqs[0])
        exit(0)

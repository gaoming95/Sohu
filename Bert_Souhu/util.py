import pickle
import os
import random
import logging
import numpy as np


class InputHelper(object):
    def read_corpus(self, datadir, input_file):
        corpus_path = os.path.join(datadir, input_file)
        sentence, tags = [], []
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
                    sentence.append(sent_)
                    tags.append(tag_)
                sent_, tag_ = [], []
        return sentence, tags

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
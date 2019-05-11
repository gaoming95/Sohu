import json
import re
import pickle

# 繁简转换会破坏某一些名词
# from langconv import Converter

# def simple2tradition(line):
#     # 将简体转换成繁体
#     line = Converter('zh-hans').convert(line)
#     return line

httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\-。%《*》/•、&＆(—)（+）：？!！“”·]+', re.UNICODE)
space = re.compile(r' +')
link = re.compile(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
repeat = re.compile(r'(.)\1{5,}')


def clean(raw):
    raw = httpcom.sub('', raw)
    raw = fil.sub('', raw)
    raw = space.sub(' ', raw)
    raw = link.sub('', raw)
    raw = repeat.sub('', raw)
    raw = raw.replace('...', '。').replace('！ ！ ！', '！').replace('！ 。', '！').replace('？ 。', '？')
    return raw


def clean_entity(keyword):
    keyword = str(keyword).replace('《', '').replace('》', '')
    return keyword


def build_dict():
    with open('./data/coreEntityEmotion_train.txt', 'r', encoding='utf-8') as f:
        train_data = f.readlines()

    with open('./data/coreEntityEmotion_test_stage1.txt', 'r', encoding='utf-8') as f:
        valid_data = f.readlines()

    train_g = open('./data/train', 'w', encoding='utf-8')
    valid_g = open('./data/test', 'w', encoding='utf-8')

    train = [(clean(json.loads(line)['title']), clean(json.loads(line)['content']),
              json.loads(line)['coreEntityEmotions']) for line in train_data]
    valid = [(clean(json.loads(line)['title']), clean(json.loads(line)['content'])) for line in valid_data]

    words = []
    print('train')
    for i in range(len(train)):
        line = train[i][0].strip() + '。' + train[i][1].strip()
        temp = [con for con in list(line)]
        train_sents = ' '.join(temp)
        train_keys = ' '.join(list(','.join([clean_entity(index['entity']) for index in train[i][2]])))
        train_g.write(train_sents + '\t' + train_keys + '\n')
        words += temp

    print("test")
    for i in range(len(valid)):
        line = valid[i][0] + valid[i][1]
        temp = [con for con in list(line)]

        valid_save = ' '.join(temp)
        valid_g.write(valid_save + '\n')
        words += temp
    words = set(words)
    # word_counter = collections.Counter(words).most_common()
    word_dict = {}
    word_dict["<padding>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<s>"] = 2
    word_dict["</s>"] = 3
    for word in words:
        word_dict[word] = len(word_dict)
    print(word_dict)
    with open("./data/vocab.pkl", "wb") as f:
        pickle.dump(word_dict, f)

def train_test_split_v():
    with open('./data/train', 'r', encoding='utf-8') as g:
        data = g.readlines()

    train = [line for line in data]

    from sklearn.model_selection import train_test_split

    x1, x2 = train_test_split(train, test_size=0.2)

    g1 = open('./data/train_1', 'w', encoding='utf-8')
    g2 = open('./data/valid_1', 'w', encoding='utf-8')

    for x_1 in x1:
        g1.write(x_1)
    for x_2 in x2:
        g2.write(x_2)

if __name__ == '__main__':
    build_dict()
    train_test_split_v()

import json
import re

# 全角转半角
def q_to_b(q_str):
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        b_str += chr(inside_code)
    return b_str


# 清洗字符串
httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配连接
fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\-。%《*》•、&＆(—)（+）：？!！“”·]+', re.UNICODE) # 匹配非法字符
space = re.compile(r' +') # 将一个以上的空格替换成一个空格
link = re.compile(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+') # 匹配网址
repeat = re.compile(r'(.)\1{5,}') # 超过6个以上的连续字符匹配掉比如......，人人人人人人


def clean(raw):
    raw = q_to_b(raw)
    raw = httpcom.sub('', raw)
    raw = fil.sub('', raw)
    raw = space.sub(' ', raw)
    raw = link.sub('', raw)
    raw = repeat.sub('', raw)
    raw = raw.replace('...', '。').replace('！ ！ ！', '！').replace('！ 。', '！').replace('？ 。', '？')
    return raw


def getType(type):
    if type == 'E':
        return 'E'


def split(text):
    """以标签数据分割成list"""
    res = []
    start = 0
    end = 0
    while end < len(text):
        if text[end] == '<':
            # < 前面的信息写入
            if start != end:
                res.append(text[start: end])
                start = end + 1
            else:
                start += 1
            # <>中的信息
            end = go(text, start)
            res.append(text[start: end])
            start = end + 1
            end = start
        else:
            end += 1
    if start != end:
        res.append(text[start: end])
    return res


def go(text, i):
    while i < len(text):
        if text[i] == '>':
            break
        else:
            i += 1
    return i


def text2char_ner_bio_format(text):
    segment = list(text)
    stack = []
    features = []
    start = 0
    end = len(segment)
    label = ''
    first = 0  # first表示当前的字符是否是第一个字符
    while start < end:
        if segment[start] == '<':
            tag = ''
            start += 1
            if segment[start] != '/':
                while start < end and segment[start] != '>':
                    tag += segment[start]
                    start += 1
                stack.append(getType(tag))
            else:
                start += 1
                while start < end and segment[start] != '>':
                    start += 1
                    tag += segment[start]
                stack.pop()
            first = 1
        else:
            if len(stack) == 0:
                label = 'O'
            else:
                if first == 1:
                    label = 'B-' + stack[-1]
                    first = 0
                else:
                    label = 'I-' + stack[-1]
            features.append([segment[start], label])
        start += 1
    return features


def build_train_data():
    with open('./data/coreEntityEmotion_train.txt', 'r', encoding='utf-8') as f:
        train = f.readlines()
    g_data = open('./data/train', 'a', encoding='utf-8')
    for line in train:
        line = json.loads(line)
        title = line['title'].strip()
        # 按照length排序，主要是华为、华为P30，先匹配小的，再匹配大的。不会出现嵌套
        entities = sorted([entitiesEmotions['entity'].strip() for entitiesEmotions in line['coreEntityEmotions']],
                          key=lambda x: len(x), reverse=True)
        contents = line['content'].strip()
        contents = clean(title) + '。' + clean(contents)
        for entity in entities:
            contents = contents.replace(entity, '<E>{}</E>'.format(entity))
        contents = re.split(r'[；。！!？?]+', contents)
        # 会出现一些嵌套错误，比如华为、华为P30。
        # 对于句子里面没有出现过重要实体的，去掉
        for content in contents:
            if content.__contains__('</E>'):
                try:
                    features = text2char_ner_bio_format(content)
                    for feature in features:
                        g_data.write(feature[0] + '\t' + feature[1] + '\n')
                    g_data.write('\n')
                except IndexError:
                    continue


def build_test_data():
    with open('./data/coreEntityEmotion_test_stage1.txt', 'r', encoding='utf-8') as f:
        train = f.readlines()
    g_data = open('./data/test', 'w', encoding='utf-8')
    for line in train:
        line = json.loads(line)
        title = line['title'].strip()
        contents = line['content'].strip()
        contents = clean(title) + '。' + clean(contents)
        g_data.write(contents.strip() + '\n')

if __name__ == '__main__':
    build_train_data()
    build_test_data()
## 面向脚本编程

from kashgari.tasks.seq_labeling import BLSTMCRFModel

from util import InputHelper

# 读取模型
new_model = BLSTMCRFModel.load_model('./model')

# 读取测试集数据
with open('./data/test', 'r', encoding='utf-8') as g:
    test_data = g.readlines()

# 对测试集进行预测
with open('./keywords_test', 'w', encoding='utf-8') as g_key:
    for ids, line in enumerate(test_data):
        try:
            label = InputHelper().iob_iobes(new_model.predict(line.replace('\t', '。')))
            result = InputHelper().result_to_json(line, label)
            line_keys = [entity['word'] for entity in result['entities']]
            g_key.write(','.join(line_keys) + '\n')
        except Exception as e:
            g_key.write('\n')

# “人工智能”
# 清洗一些看得到的错误，取前三个。这里做的很糙
with open('./keywords_test', 'r', encoding='utf-8') as g:
    data = g.readlines()
with open('./keywords', 'w', encoding='utf-8') as f:
    for line in data:
        line = line.replace('\t', ',').replace('	',',')
        line = line.split(',')
        temp = []
        for index in line:
            if len(index.strip())==1:
                continue
            if len(index.strip()) == 0:
                continue
            if index.strip().__contains__('“') and not index.strip().__contains__('”'):
                continue
            if index.strip().__contains__('”') and not index.strip().__contains__('“'):
                continue
            if index.strip().__contains__('vivox27'):
                index = index.replace('vivox27', 'vivo x27')
            if index not in temp:
                temp.append(index.strip())
        f.write(','.join(list(set(temp[:3]))) + '\n')
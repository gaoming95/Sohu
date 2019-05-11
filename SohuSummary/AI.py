import pickle

with open('./result.txt', 'r', encoding='utf-8') as g:
    data = g.readlines()
f = open('./keywords.txt', 'w', encoding='utf-8')
for id,line in enumerate(data):
    if(len(line)>100):
        print(id,line)
    line = line.strip().replace('<unk>','.')
    res = ''
    for i in range(0, len(line), 2):
        res += line[i]
    res = ','.join(set([r.strip() for r in res.split(',')]))
    f.write(res + '\n')

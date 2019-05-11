# import tensorflow as tf

#
# x = [[1, 2, 3], [2, 3, 4]]
# x_mask = tf.cast(tf.greater(tf.expand_dims(x, 2), 0), tf.float32)
# print(x_mask.shape)
# x = tf.cast(x, tf.int32)
# x = tf.one_hot(x, depth=6, axis=-1)
# x = tf.reduce_sum(tf.multiply(x_mask, x), 1, keep_dims=True)
# x = tf.cast(tf.greater(x, 0.5), tf.float32)
# x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[-1]])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a0 = sess.run(x)
#     print("var0(axis=0 depth=3)\n", a0)
# import numpy as np
# x = [1, 2, 3]
# y = [2, 3, 4]
# test_res = []
# z = tf.reduce_sum(tf.add(x, y))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a0 = sess.run(z)
#     a1 = sess.run(z)
#     test_res.append(a0)
#     test_res.append(a1)
#     print(np.average(test_res))
#     print(type(test_res))
# input_shape = (2, 1, 10)
# kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
# print(kernel_shape)

# a = [[1, 2, 3], [2, 3, 4]]
# b = [1, 2, 3]
# # c = tf.cast([1, 1, 1],tf.int32)
# c = [1, 1, 1]
# d = tf.add(tf.multiply(a, b), c)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a0 = sess.run(d)
#     print(a0)

# import matplotlib.pyplot as plt
#
# with open('./data/train', 'r', encoding='utf-8') as g:
#     data = g.readlines()
# len_ = []
# for line in data:
#     len_.append(len(line.split('\t')[0].strip().split(' ')))
# plt.hist(len_, bins=30)
# plt.show()

import numpy as np

# a = [[[7, 0, 7, 1], [3, 5, 9, 1], [8, 5, 1, 1]], [[7, 0, 7, 1], [3, 5, 9, 1], [8, 5, 1, 1]]]
# b = [[1, 2, 0, 0], [1, 3, 4, 6]]
# c = tf.add(a, tf.expand_dims(b,1))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     x0 = sess.run(c)
#     print(x0)

#
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


# def get_common_len(list1, list2):
#     """得到共同序列的长度"""
#     common_len = 0
#     for i in range(min(len(list1), len(list2))):
#         if list1[i] != list2[i]:
#             break
#         common_len += 1
#     return common_len
#
#
# def compute_max_sub_list(name_list):
#     # 1. 得到所有的后缀数组
#     suffix_sub_list = []
#     for i in range(len(name_list)):
#         suffix_sub_list.append(name_list[i:])
#
#     # 2.后缀子数组 排序
#     suffix_sub_list.sort()
#
#     # 3.求相邻的后缀子数组 的最长公共数组
#     max_suffix_sub = []
#     for i in range(len(suffix_sub_list) - 1):
#         common_len = get_common_len(suffix_sub_list[i], suffix_sub_list[i + 1])
#         if common_len <= len(max_suffix_sub):
#             continue
#         max_suffix_sub = suffix_sub_list[i][:common_len]
#     return ''.join(max_suffix_sub)

# string = 'x858885885885885885885885885885885885885885885885885'
# res = compute_max_sub_list(list(string))
# print(res)
# print(string.replace(res,''))

# import pickle
# with open('./data/vocab.pkl','rb') as f:
#     data = pickle.load(f)
# print(data)
# print(data['3'])

# string = 'i p h o n e   x s \n'
# res = [string[i] for i in range(0,len(string),2)]
# print(res)
# print(string.split(' '))
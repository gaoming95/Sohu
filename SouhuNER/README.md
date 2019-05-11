# SouhuNER

传统的NER模型，没有BERT，效果很差。实体应该是0.43左右，没有优化

写了验证集的评估，自己做的时候也做了，但是上传的时候把数据脚本删了，就没有在写了。
我的处理方式是用sklearn的train_test_split把原始数据集拆分，然后在render.py里面分别传进去，得到train和valid。

bert.py 可以用bert as service做bert的pretrain，而不是fine-tuning

conlleval.py 评估结果conll官方文件

main.py 主函数

model.py 模型文件

play.py 测试集输出结果

render.py 处理数据

util.py 工具包

# 使用方式

（1）新建data文件将原始数据放进去

（2）python render.py 处理数据

（3）python main.py 运行模型

（4）python play.py 对测试数据进行测试

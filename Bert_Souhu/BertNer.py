from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from util import InputHelper
train_x, train_y = InputHelper().read_corpus('data', 'Bert_train')
embedding = BERTEmbedding('./chinese_L-12_H-768_A-12', sequence_length=256)
model = BLSTMCRFModel(embedding)
model.fit(train_x,
          train_y,
          epochs=10,
          batch_size=512)
model.save('./model')

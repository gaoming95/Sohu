import pickle
from bert_serving.client import BertClient


def embed():
    bc = BertClient()
    with open('data/vocab.pkl', 'rb') as fr:
        dictionary = pickle.load(fr)
    dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=False)
    word_embeddings = [key for key, _ in dictionary]
    embedding = bc.encode(word_embeddings)
    with open('data/embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)


def test():
    with open('data/embedding.pkl', 'rb') as fr:
        dictionary = pickle.load(fr)
    print(dictionary[0])
test()
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# テキストデータのサンプル
text_data = [
    "Word embeddings are a type of word representation.",
    "They allow words to be represented as vectors in a continuous vector space.",
    "These vectors have the property that words that are closer in the vector space are expected to be similar in meaning.",
    "This allows us to perform various operations such as vector arithmetic on words."
]

# テキストデータをトークン化してリストに変換
tokenized_text = [word_tokenize(sentence.lower()) for sentence in text_data]

# Word2Vecモデルの学習
model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# 単語のベクトルを取得
word_vectors = model.wv

# "word" のベクトルを取得
word_vector = word_vectors['word']
print("Vector representation of 'word':", word_vector)
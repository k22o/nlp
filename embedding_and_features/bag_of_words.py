from sklearn.feature_extraction.text import CountVectorizer

# テキストデータのサンプル
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# CountVectorizerを使ってBoWを作成
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)

# BoWの各単語のインデックスを表示
print("Vocabulary:", vectorizer.get_feature_names_out())

# 各文書のBoW表現を表示
print("BoW Representation:")
print(bow.toarray())
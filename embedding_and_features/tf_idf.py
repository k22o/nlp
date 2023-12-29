from sklearn.feature_extraction.text import TfidfVectorizer

# テキストデータのサンプル
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# TfidfVectorizerを使ってTF-IDFを計算
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# 各単語のインデックスを表示
print("Vocabulary:", vectorizer.get_feature_names_out())

# TF-IDF表現を表示
print("TF-IDF Representation:")
print(tfidf_matrix.toarray())
from nltk.translate.bleu_score import sentence_bleu

reference_text = ["This is a sample reference sentence."]
prediction_text = "This is a sample prediction sentence."

# 参照文と予測文をトークン化（単語やフレーズなどに分割）してBLEUスコアを計算
reference_tokens = [sentence.split() for sentence in reference_text]
prediction_tokens = prediction_text.split()

# BLEUスコアを計算（weightsはNグラムの重みを指定）
bleu_score = sentence_bleu(reference_tokens, prediction_tokens, weights=(0.25, 0.25, 0.25, 0.25))
print("BLEU Score:", bleu_score)
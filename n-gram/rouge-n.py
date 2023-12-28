from nltk.util import ngrams
from pprint import pprint

def rouge_n(reference,prediction, n=1):
    reference_ngrams = list(ngrams(reference, n))
    prediction_ngrams = list(ngrams(prediction, n))
    
    # 共通のNグラムの件数
    count_common = len(set(reference_ngrams).intersection(prediction_ngrams))
    # referenceのNグラムの件数
    count_reference = len(reference_ngrams)
    # predictionのNグラムの件数
    count_prediction = len(prediction_ngrams)

    # スコア計算
    precision = 0.0
    recall = 0.0
    f_value = 0.0

    # ここが大事なところ！
    if count_reference != 0:
        precision = count_common / count_prediction;
        recall = count_common / count_reference

    if precision + recall != 0:    
        f_value = 2*precision*recall/(precision + recall)

    return {
        "precisioin": precision,
        "recall": recall,
        "f-value": f_value
    }

reference_text = "This is a sample reference sentence."
prediction_text = "This is a sample hypothesis sentence."

rouge_1_score = rouge_n(reference_text.split(), prediction_text.split(), n=1)
pprint(rouge_1_score);

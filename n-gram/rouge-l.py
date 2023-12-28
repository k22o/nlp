from pprint import pprint

def lcs(reference, prediction):
    m = len(reference)
    n = len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# ROUGE-Lの近似的な計算
def rouge_l(reference, prediction):

    # 参照文と生成文の最長共通部分列の長さを取得し、ROUGE-Lを計算
    lcs_length = lcs(reference, prediction)
    reference_length = len(reference)
    prediction_length = len(prediction)
    
    precision = 0.0
    recall = 0.0
    f_value = 0.0

    if reference_length != 0:
        precision = lcs_length / prediction_length
        recall = lcs_length / reference_length

    if precision + recall != 0:    
        f_value = 2*precision*recall/(precision + recall)
    
    return {
        "precisioin": precision,
        "recall": recall,
        "f-value": f_value
    }

reference_text = "This is a sample reference sentence."
prediction_text = "This is a sample prediction sentence."

rouge_l_score = rouge_l(reference_text.split(), prediction_text.split())
pprint(rouge_l_score)

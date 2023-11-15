from rouge_chinese import Rouge
import jieba


def calculate_chinese_rouge_scores(hypothesis, reference):
    hypothesis = " ".join(jieba.cut(hypothesis))
    reference = " ".join(jieba.cut(reference))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

if __name__ == "__main__":
    pass
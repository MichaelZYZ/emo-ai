from transformers import pipeline
import re

class TextEmotionRecognizer:
    def __init__(self):
        # 英文多类别情绪模型
        self.en_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
        # 中文三分类情绪模型
        self.cn_classifier = pipeline('sentiment-analysis', model='uer/roberta-base-finetuned-dianping-chinese')

    def is_chinese(self, text):
        return re.search('[\u4e00-\u9fff]', text) is not None

    def predict(self, text):
        if self.is_chinese(text):
            result = self.cn_classifier(text)[0]
            # dianping-chinese 输出: positive/negative/neutral
            label_map = {'positive': '开心', 'negative': '愤怒', 'neutral': '中性'}
            label = result['label'].lower()
            return label_map.get(label, label)
        else:
            # 英文多类别情绪
            results = self.en_classifier(text)[0]
            # 取置信度最高的类别
            best = max(results, key=lambda x: x['score'])
            label = best['label'].lower()
            # 英文多类别映射
            label_map = {
                'joy': '开心',
                'anger': '愤怒',
                'sadness': '悲伤',
                'fear': '恐惧',
                'surprise': '惊讶',
                'disgust': '厌恶',
                'neutral': '中性'
            }
            return label_map.get(label, label)

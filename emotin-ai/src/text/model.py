import re
from collections import Counter

from transformers import pipeline

class TextEmotionRecognizer:
    def __init__(self):
        self.en_classifier = None
        self.cn_classifier = None
        self.cn_rules = {
            '开心': ['开心', '高兴', '愉快', '快乐', '幸福', '兴奋'],
            '愤怒': ['生气', '愤怒', '气死', '火大', '恼火'],
            '悲伤': ['伤心', '难过', '悲伤', '沮丧', '失落', '哭'],
            '恐惧': ['害怕', '恐惧', '紧张', '担心', '焦虑'],
            '惊讶': ['惊讶', '震惊', '意外', '吃惊'],
            '厌恶': ['恶心', '讨厌', '厌恶', '反感']
        }
        self.en_rules = {
            '开心': ['happy', 'joy', 'excited', 'glad', 'great', 'awesome'],
            '愤怒': ['angry', 'mad', 'furious', 'annoyed', 'rage'],
            '悲伤': ['sad', 'down', 'depressed', 'upset', 'cry'],
            '恐惧': ['afraid', 'fear', 'scared', 'anxious', 'nervous'],
            '惊讶': ['surprised', 'shock', 'amazed', 'unexpected'],
            '厌恶': ['disgust', 'gross', 'nasty', 'hate']
        }
        self._init_models()

    def _init_models(self):
        try:
            # 英文多类别情绪模型（优先本地缓存，避免无网环境报错）
            self.en_classifier = pipeline(
                'text-classification',
                model='j-hartmann/emotion-english-distilroberta-base',
                return_all_scores=True,
                local_files_only=True
            )
        except Exception:
            self.en_classifier = None

        try:
            # 中文三分类情绪模型（优先本地缓存）
            self.cn_classifier = pipeline(
                'sentiment-analysis',
                model='uer/roberta-base-finetuned-dianping-chinese',
                local_files_only=True
            )
        except Exception:
            self.cn_classifier = None

    def is_chinese(self, text):
        return re.search('[\u4e00-\u9fff]', text) is not None

    def _rule_predict(self, text, rules):
        text_lower = text.lower()
        hit_counter = Counter()
        for emotion, keywords in rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    hit_counter[emotion] += 1
        if hit_counter:
            return hit_counter.most_common(1)[0][0]
        return '中性'

    def predict(self, text):
        if self.is_chinese(text):
            if self.cn_classifier is not None:
                result = self.cn_classifier(text)[0]
                # dianping-chinese 输出: positive/negative/neutral
                label_map = {'positive': '开心', 'negative': '愤怒', 'neutral': '中性'}
                label = result['label'].lower()
                return label_map.get(label, label)
            return self._rule_predict(text, self.cn_rules)
        else:
            if self.en_classifier is not None:
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
            return self._rule_predict(text, self.en_rules)

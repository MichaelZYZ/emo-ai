import numpy as np
import os
import random
try:
    from pyAudioAnalysis import audioClassification
except ImportError:
    audioClassification = None

class AudioEmotionRecognizer:
    def __init__(self):
        # 你可以用 pyAudioAnalysis 训练自己的模型并替换下方路径
        self.model_path = 'pyAudioAnalysis_model'  # 需提前训练好模型
        self.labels = ['angry', 'happy', 'neutral', 'sad', 'surprise', 'fear']

    def predict(self, audio_path):
        if audioClassification and os.path.exists(self.model_path):
            # 使用 pyAudioAnalysis 进行音频情感分类
            [Result, P, classNames] = audioClassification.file_classification(audio_path, self.model_path, 'svm')
            # classNames 是标签名，Result 是预测类别索引
            return classNames[int(Result)]
        else:
            # 未安装pyAudioAnalysis或无模型时，随机返回
            return random.choice(self.labels)

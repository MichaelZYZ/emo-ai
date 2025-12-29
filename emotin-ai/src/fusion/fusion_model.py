from collections import Counter

class FusionModel:
    def __init__(self):
        pass

    def fuse(self, emotion_list):
        # 简单投票法
        count = Counter(emotion_list)
        return count.most_common(1)[0][0]

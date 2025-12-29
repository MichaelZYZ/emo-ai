from src.text.model import TextEmotionRecognizer
from src.audio.model import AudioEmotionRecognizer
from src.image.model import ImageEmotionRecognizer
from src.fusion.fusion_model import FusionModel

def main(text, audio_path, image_path):
    text_model = TextEmotionRecognizer()
    audio_model = AudioEmotionRecognizer()
    image_model = ImageEmotionRecognizer()
    fusion_model = FusionModel()

    text_emotion = text_model.predict(text)
    print(f"文本情绪识别结果: {text_emotion}")
    audio_emotion = audio_model.predict(audio_path)
    print(f"语音情绪识别结果: {audio_emotion}")
    image_emotion = image_model.predict(image_path)
    print(f"表情情绪识别结果: {image_emotion}")

    final_emotion = fusion_model.fuse([text_emotion, audio_emotion, image_emotion])

    # 判断主语言（文本优先，只有文本为英文且语音也为英文才输出英文）
    import re
    def is_chinese(s):
        return re.search('[\u4e00-\u9fff]', s) is not None

    if is_chinese(text):
        print(f"综合情绪分析结果：{final_emotion}")
    else:
        # 英文输出映射
        en_map = {'开心': 'POSITIVE', '愤怒': 'NEGATIVE', '中性': 'NEUTRAL', '悲伤': 'SAD', '惊讶': 'SURPRISE', '恐惧': 'FEAR'}
        print(f"Final emotion analysis: {en_map.get(final_emotion, final_emotion)}")

if __name__ == "__main__":
    # 示例输入
    main("我很伤心", "weather_forecast.wav", "smile.jpg")

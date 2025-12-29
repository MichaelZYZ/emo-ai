
import numpy as np
import cv2
import os

from src.image.my_model import MyImageEmotionRecognizer

class ImageEmotionRecognizer:
    def __init__(self, use_custom=False, custom_model_path='my_emotion_model.pth'):
        self.use_custom = use_custom
        if use_custom:
            self.model = MyImageEmotionRecognizer(model_path=custom_model_path)
        else:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.cn_map = {
                'angry': '愤怒',
                'disgust': '厌恶',
                'fear': '恐惧',
                'happy': '开心',
                'sad': '悲伤',
                'surprise': '惊讶',
                'neutral': '中性'
            }

    def detect_and_crop_face(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = img[y:y+h, x:x+w]
            temp_path = 'cropped_face.jpg'
            cv2.imwrite(temp_path, face_img)
            return temp_path
        else:
            return image_path

    def predict(self, image_path):
        if hasattr(self, 'model'):
            # 使用自定义PyTorch模型
            return self.model.predict(image_path)
        try:
            # deepface/FER逻辑
            cropped_path = self.detect_and_crop_face(image_path)
            result = self.DeepFace.analyze(img_path=cropped_path, actions=['emotion'], enforce_detection=True, detector_backend='opencv', models={'emotion': self.DeepFace.build_model('Emotion')}, prog_bar=False)
            emotion = result['dominant_emotion']
            emotion_scores = result['emotion']
            print('图片情绪概率分布:', {self.cn_map.get(k, k): round(v, 2) for k, v in emotion_scores.items()})
            if cropped_path != image_path and os.path.exists(cropped_path):
                os.remove(cropped_path)
            return self.cn_map.get(emotion, emotion)
        except Exception as e:
            print('图片情绪识别异常:', e)
            return np.random.choice(list(self.cn_map.values()))

import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import shutil

from src.text.model import TextEmotionRecognizer
from src.audio.model import AudioEmotionRecognizer
from src.image.model import ImageEmotionRecognizer
from src.fusion.fusion_model import FusionModel

templates = Jinja2Templates(directory="web/templates")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(
    text: str = Form(""),
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):
    # 保存上传文件
    audio_path, image_path = None, None
    if audio:
        audio_path = f"web/tmp_{audio.filename}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    if image:
        image_path = f"web/tmp_{image.filename}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    try:
        text_model = TextEmotionRecognizer()
        audio_model = AudioEmotionRecognizer()
        image_model = ImageEmotionRecognizer()
        fusion_model = FusionModel()
        text_emotion = text_model.predict(text) if text else None
        audio_emotion = audio_model.predict(audio_path) if audio_path else None
        image_emotion = image_model.predict(image_path) if image_path else None
        result_list = [x for x in [text_emotion, audio_emotion, image_emotion] if x]
        final_emotion = fusion_model.fuse(result_list) if result_list else "无分析结果"
        # 判断主语言
        import re
        def is_chinese(s):
            return s and re.search('[\u4e00-\u9fff]', s) is not None
        if is_chinese(text):
            result = f"综合情绪分析结果：{final_emotion}"
        else:
            en_map = {'开心': 'POSITIVE', '愤怒': 'NEGATIVE', '中性': 'NEUTRAL', '悲伤': 'SAD', '惊讶': 'SURPRISE', '恐惧': 'FEAR'}
            result = f"Final emotion analysis: {en_map.get(final_emotion, final_emotion)}"
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)})
    finally:
        # 清理临时文件
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

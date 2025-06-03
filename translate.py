from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from flask import jsonify
from transformers import MarianMTModel, MarianTokenizer
import os, base64
from io import BytesIO

# 创建独立虚拟环境: python -m venv venv
# 激活虚拟环境: venv\Scripts\activate

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 初始化 OCR（支持中英文日文）
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def image_to_base64(img_path):
    img = Image.open(img_path).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 支持的语言对及其模型
MODEL_MAP = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-ja": "Helsinki-NLP/opus-mt-zh-ja",
    "ja-zh": "Helsinki-NLP/opus-mt-ja-zh",
    "zh-ko": "Helsinki-NLP/opus-mt-zh-ko",
    "ko-zh": "Helsinki-NLP/opus-mt-ko-zh",
    "zh-de": "Helsinki-NLP/opus-mt-zh-de",
    "de-zh": "Helsinki-NLP/opus-mt-de-zh",
}

# 模型缓存
models = {}
tokenizers = {}

def load_model(model_name):
    if model_name not in models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model
    return tokenizers[model_name], models[model_name]


def translate_text(text: str, src_lang: str, tgt_lang: str):

    lang_key = f"{src_lang}-{tgt_lang}"
    model_name = MODEL_MAP.get(lang_key)

    if not model_name:
        return jsonify({"error": f"Translation from {src_lang} to {tgt_lang} is not supported."}), 400

    try:
        tokenizer, model = load_model(model_name)
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, max_length=512)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        print(translated_text)
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "translated_text": translated_text,
        "source_language": src_lang,
        "target_language": tgt_lang
    })


def translate_image(image_path: str, src_lang: str, tgt_lang: str):

    lang_key = f"{src_lang}-{tgt_lang}"
    model_name = MODEL_MAP.get(lang_key)
    if not model_name:
        return jsonify({"error": f"Unsupported language pair: {lang_key}"}), 400

    # OCR识别
    try:
        result = ocr.predict(input=image_path)
        # 识别结果处理，提取文字和位置，翻译文字
        translated_results = []
        result_0 = result[0]
        if result_0 :
            lang_key = f"{src_lang}-{tgt_lang}"
            model_name = MODEL_MAP.get(lang_key)

            rec_texts = result_0.get("rec_texts", [])
            rec_polys = result_0.get("rec_polys", [])
            rec_scores = result_0.get("rec_scores", [])

            # 画图
            original_img = Image.open(image_path).convert("RGB")
            # 画布
            draw = ImageDraw.Draw(original_img)
            font_size = 20
            try:
                # C:\Windows\Fonts
                font = ImageFont.truetype("../simfang.ttf", font_size)
            except:
                font = ImageFont.load_default()

            for orig_text, poly, score in zip(rec_texts, rec_polys, rec_scores):
                print(f"orig_text: {orig_text}")
                try:
                    tokenizer, model = load_model(model_name)
                    inputs = tokenizer([orig_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
                    translated = model.generate(**inputs, max_length=512)
                    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    print(f"translated_text: {translated_text}")
                    print()
                    translated_results.append({
                        'poly': poly,
                        'original': orig_text,
                        'translated': translated_text
                    })
                    # 画多边形（线条）
                    draw.polygon(poly, outline="red", width=1)
                    # 画文字（在第一个顶点）
                    draw.text((poly[0][0], poly[0][1] - 15), translated_text, fill="blue", font=font)
                except Exception as e:
                    print(str(e))

            # 显示或保存
            original_img.show()
            # 转 base64 返回
            buffered = BytesIO()
            original_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return jsonify({
                "image_base64": img_base64
            })

        return jsonify({
            "image_base64": ""
        })

    except Exception as e:
        print(str(e))
        return jsonify({"error": f"OCR error: {str(e)}"}), 500

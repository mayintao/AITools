import uuid

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import os, base64
from io import BytesIO

# 创建独立虚拟环境: python -m venv venv
# 激活虚拟环境: venv\Scripts\activate

app = Flask(__name__)
CORS(app)

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


@app.route("/translate_text", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text", "")
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")

    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "text, src_lang, tgt_lang are required"}), 400

    lang_key = f"{src_lang}-{tgt_lang}"
    model_name = MODEL_MAP.get(lang_key)

    if not model_name:
        return jsonify({"error": f"Translation from {src_lang} to {tgt_lang} is not supported."}), 400

    try:
        tokenizer, model = load_model(model_name)
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs, max_length=256)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "translated_text": translated_text,
        "source_language": src_lang,
        "target_language": tgt_lang
    })


@app.route("/translate_image", methods=["POST"])
def translate_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    src_lang = request.form.get("src_lang")
    tgt_lang = request.form.get("tgt_lang")

    if not src_lang or not tgt_lang:
        return jsonify({"error": "src_lang and tgt_lang are required"}), 400

    lang_key = f"{src_lang}-{tgt_lang}"
    model_name = MODEL_MAP.get(lang_key)
    if not model_name:
        return jsonify({"error": f"Unsupported language pair: {lang_key}"}), 400

    # 保存上传图片
    file = request.files["image"]
    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # OCR识别
    try:
        result = ocr.predict(input=filepath)
        first_image_result = result[0]  # 👈 获取第一张图的所有识别块
        texts = first_image_result['rec_texts']
        full_text = "\n".join(texts)
        print(full_text)
    except Exception as e:
        return jsonify({"error": f"OCR error: {str(e)}"}), 500

    # 翻译
    try:
        tokenizer, model = load_model(model_name)
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs, max_length=512)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    except Exception as e:
        return jsonify({"error": f"Translation error: {str(e)}"}), 500

    # 创建拼接图
    try:
        original_img = Image.open(filepath).convert("RGB")
        orig_w, orig_h = original_img.size

        # 新图大小：宽 = 原图宽 + 右边空白区域；高 = 原图高
        padding = 50
        new_w = orig_w + int(orig_w * 0.8)
        new_h = orig_h

        new_img = Image.new("RGB", (new_w, new_h), color=(255, 255, 255))
        new_img.paste(original_img, (0, 0))

        # 在右边绘制翻译文字
        draw = ImageDraw.Draw(new_img)
        font_size = 20
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 自动换行写入翻译文本
        from textwrap import wrap
        line_width = int(orig_w * 0.7 / font_size)
        lines = wrap(translated_text, width=line_width)

        x_offset = orig_w + padding
        y_offset = padding

        for line in lines:
            draw.text((x_offset, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += font_size + 6

        # 转 base64 返回
        buffered = BytesIO()
        new_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        return jsonify({"error": f"Translation or drawing error: {str(e)}"}), 500

    return jsonify({
        "ocr_text": full_text,
        "translated_text": translated_text,
        "image_base64": img_base64,
        "source_language": src_lang,
        "target_language": tgt_lang
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

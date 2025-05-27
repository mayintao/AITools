import io
from PIL import Image
import easyocr
import torch
from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
CORS(app)

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
reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

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

    image = request.files["image"]
    image = Image.open(image.stream).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    try:
        result = reader.readtext(image_bytes, detail=0)
        full_text = " ".join(result)

        tokenizer, model = load_model(model_name)
        inputs = tokenizer([full_text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs, max_length=256)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "ocr_text": full_text,
        "translated_text": translated_text,
        "source_language": src_lang,
        "target_language": tgt_lang
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

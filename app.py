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

def load_model(model_name):
    if model_name not in models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model
    return tokenizers[model_name], models[model_name]

@app.route("/text_translate", methods=["POST"])
def text_translate():
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

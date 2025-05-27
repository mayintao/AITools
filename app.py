from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
print(f"Loading model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded.")
print(tokenizer.lang_code_to_id.keys())

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")

    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "text, src_lang, tgt_lang are required"}), 400

    if src_lang not in tokenizer.lang_code_to_id or tgt_lang not in tokenizer.lang_code_to_id:
        return jsonify({"error": "Unsupported source or target language"}), 400

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")

    try:
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_new_tokens=256
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        logging.exception("Translation error")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "translated_text": translated_text,
        "source_language": src_lang,
        "target_language": tgt_lang
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

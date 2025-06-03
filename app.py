from flask import Flask, request, jsonify

from translate.translate import translate_text
from translate.translate import translate_image
from flask_cors import CORS
import os, uuid

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)


# 文字翻译
@app.route("/ai/api_translate_text", methods=["POST"])
def api_translate_text():
    data = request.get_json()
    text = data.get("text", "")
    print(text)
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")

    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "text, src_lang, tgt_lang are required"}), 400

    try:
        results = translate_text(text, src_lang, tgt_lang)
        print(results)
        return results
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 图片翻译
@app.route("/ai/api_translate_image", methods=["POST"])
def api_translate_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    src_lang = request.form.get("src_lang")
    tgt_lang = request.form.get("tgt_lang")
    if not src_lang or not tgt_lang:
        return jsonify({"error": "src_lang, tgt_lang are required"}), 400
    file.save(filepath)
    result = translate_image(filepath, src_lang, tgt_lang)
    # 清理上传图
    if os.path.exists(filepath):
        os.remove(filepath)
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify

from translate.translate import translate_text
from translate.translate import translate_image
from flask_cors import CORS
import os, uuid

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)


@app.route("/ai/api_ocr_image", methods=["POST"])
def api_ocr_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    result = ocr_image(filepath)
    # 清理上传图
    if os.path.exists(filepath):
        os.remove(filepath)
    return result

# 将输入文本改写为圣经风格的表达方式
@app.route("/ai/api_bible_paraphrase", methods=["POST"])
def api_bible_paraphrase():
    data = request.get_json()
    prompt = data.get("prompt")
    print(prompt)
    num_return_sequences = int(data.get("num_return_sequences", 3))

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    try:
        results = generate_bible_paraphrases(prompt, num_return_sequences)
        print(results)
        return results
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# 文字生成图片
@app.route("/ai/api_generate_images", methods=["POST"])
def api_generate_images():
    data = request.get_json()
    prompt = data.get("prompt")
    print(prompt)
    num_images = int(data.get("num_images", 1))

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    try:
        results = generate_images(prompt, num_images)
        return results
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai/api_aiask_p", methods=["POST"])
def api_aiask_p():
    data = request.json
    question = data.get("question")
    answer = get_answer(question)
    return jsonify({"answer": answer})

@app.route("/ai/api_aiask_g", methods=["GET"])
def api_aiask_g():
    question = request.args.get("question")  # 从 URL 参数读取
    if not question:
        return jsonify({"error": "请提供参数 question，例如 /ask?question=蛋白质摄入量"})
    answer = get_answer(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

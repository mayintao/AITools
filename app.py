import os
# 设置 huggingface 的缓存路径（你已经设置了）
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
# 将 PaddleOCR 的模型缓存路径指向 /tmp 下
os.environ["PADDLEOCR_HOME"] = "/tmp/paddleocr_models"
os.makedirs("/tmp/paddleocr_models", exist_ok=True)

import uuid
import time
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS
from paddleocr import PaddleOCR
from flask import jsonify, Flask, request
from transformers import MarianMTModel, MarianTokenizer
import base64
from io import BytesIO
import logging
# 字典初始化时自动赋默认值
from collections import defaultdict 
import traceback

# 创建独立虚拟环境: python -m venv venv
# 激活虚拟环境: venv\Scripts\activate

app = Flask(__name__)
CORS(app)

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# 请求计数器
request_counter = defaultdict(int)
# 全局计数器
translate_image_counter = 0


# 初始化 OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_model_dir='/tmp/paddleocr_models/det',
    rec_model_dir='/tmp/paddleocr_models/rec',
    cls_model_dir='/tmp/paddleocr_models/cls'
)

def image_to_base64(img_path):
    img = Image.open(img_path).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 加载中文字体
def load_chinese_font(size=24):
    try:
        print("尝试加载中文字体 simfang.ttf")
        return ImageFont.truetype("simfang.ttf", size=size)
    except Exception as e:
        print(f"⚠️ 加载中文字体失败: {e}")
        return ImageFont.load_default()


# 支持的语言对及其模型
MODEL_MAP = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-de": "Helsinki-NLP/opus-mt-zh-de",
    "de-zh": "Helsinki-NLP/opus-mt-de-zh",
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap",
}

# 模型缓存
models = {}
tokenizers = {}


# 启动时预加载所有模型，避免每次都加载模型，特别耗时！
def preload_models():
    print("🚀 正在预加载所有翻译模型...")
    for model_name in set(MODEL_MAP.values()):
        try:
            # 即在 models 和 tokenizers 中，就直接返回缓存，不再执行 from_pretrained(...)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            tokenizers[model_name] = tokenizer
            models[model_name] = model
            print(f"✅ 模型已加载: {model_name}")
        except Exception as e:
            print(f"❌ 加载失败: {model_name} -> {e}")


def load_model(model_name):
      # 获取已缓存的模型和 tokenizer，如果不存在就加载（容错）
    if model_name not in models or model_name not in tokenizers:
        logging.warning(f"⚠️ 模型 {model_name} 未预加载，正在动态加载！")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model
    return tokenizers[model_name], models[model_name]


def translate_text(text: str, src_lang: str, tgt_lang: str):
    start_time = time.time()
    
    lang_key = f"{src_lang}-{tgt_lang}"
    request_counter[lang_key] += 1
    model_name = MODEL_MAP.get(lang_key)

    if not model_name:
        return jsonify({"error": f"Translation from {src_lang} to {tgt_lang} is not supported."}), 400

    try:
        tokenizer, model = load_model(model_name)
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, max_length=512)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        elapsed = round((time.time() - start_time) * 1000, 2)
        client_ip = request.remote_addr or "unknown"

        # 日志输出
        logging.info(
            f"[{client_ip}] {src_lang}->{tgt_lang} | \"{text[:30]}\" -> \"{translated_text[:30]}\" | {elapsed}ms | Total: {request_counter[lang_key]}"
        )
        
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "translated_text": translated_text,
        "source_language": src_lang,
        "target_language": tgt_lang
    })


def translate_image(image_path: str, src_lang: str, tgt_lang: str):
    global translate_image_counter
    translate_image_counter += 1
    request_id = uuid.uuid4().hex[:8]

    lang_key = f"{src_lang}-{tgt_lang}"
    print(lang_key)
    model_name = MODEL_MAP.get(lang_key)
    print(model_name)
    if not model_name:
        logging.warning(f"[{request_id}] ❌ 不支持的语言对: {lang_key}")
        return jsonify({"error": f"Unsupported language pair: {lang_key}"}), 400

    try:
        # ocr 识别
        print('ocr 识别 start')
        result = ocr.ocr(image_path)

        original_img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_img)

        print(f"[DEBUG] tgt_lang = '{tgt_lang}'")
        if tgt_lang.strip().lower() == 'zh':
            print('goto load_chinese_font')
            font = load_chinese_font(size=24)
        else:
            print('load_default font')
            font = ImageFont.load_default()
        
        translated_results = []
        ocr_result = result[0]

        for box, (orig_text, score) in ocr_result:
            logging.info(f"orig_text: {orig_text}")
            try:
                tokenizer, model = load_model(model_name)
                inputs = tokenizer([orig_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = model.generate(**inputs, max_length=512)
                translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                logging.info(f"[{request_id}] 🔁 翻译: {translated_text}")
        
                translated_results.append({
                    'poly': box,
                    'original': orig_text,
                    'translated': translated_text
                })
        
                draw.polygon(box, outline="red", width=1)
                draw.text((box[0][0], box[0][1] - 15), translated_text, fill="blue", font=font)
            except Exception as e:
                logging.error(f"[{request_id}] ❗ 翻译失败: {e}")
        
        buffered = BytesIO()
        original_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(f"[{request_id}] 🖼️ 图像翻译完成，成功生成 base64 图像")
        return jsonify({
            "image_base64": img_base64
        })

    except Exception as e:
        logging.error(f"[{request_id}] ❗ 整体翻译失败: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"OCR error: {str(e)}"}), 500


# 文字翻译 post
@app.route("/ai/api_translate_text", methods=["POST"])
def api_translate_text():
    
    data = request.get_json()
    text = data.get("text", "")
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")
    print(text,src_lang,tgt_lang)

    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "text, src_lang, tgt_lang are required"}), 400

    try:
        results = translate_text(text, src_lang, tgt_lang)
        print(results)
        return results
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 文字翻译 get
# https://yintao127-translate.hf.space/ai/api_translate_text_g?text=你好&src_lang=zh&tgt_lang=en
@app.route("/ai/api_translate_text_g", methods=["GET"])
def api_translate_text_g():
    
    text = request.args.get("text", "")
    src_lang = request.args.get("src_lang", "")
    tgt_lang = request.args.get("tgt_lang", "")
    print(text,src_lang,tgt_lang)

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
    logging.info(uid)
    filename = f"{uid}.jpg"
    filepath = os.path.join("/tmp", filename)
    logging.info(filepath)

    src_lang = request.form.get("src_lang")
    tgt_lang = request.form.get("tgt_lang")
    logging.info(tgt_lang)
    
    if not src_lang or not tgt_lang:
        return jsonify({"error": "src_lang, tgt_lang are required"}), 400
    file.save(filepath)
    result = translate_image(filepath, src_lang, tgt_lang)
    # 清理上传图
    if os.path.exists(filepath):
        os.remove(filepath)
    return result


# test
# https://yintao127-translate.hf.space/ai/api_test_g
@app.route("/ai/api_test_g", methods=["GET"])
def api_test_g():
    return jsonify({"test": "ok!"})


# 查看当前翻译统计
# https://yintao127-translate.hf.space/ai/api_translate_stats
@app.route("/ai/api_translate_stats", methods=["GET"])
def api_translate_stats():
    return jsonify(dict(request_counter))


# 查看总请求次数接口（用于状态页）
# https://yintao127-translate.hf.space/ai/api_translate_image_status
@app.route("/ai/api_translate_image_status", methods=["GET"])
def api_translate_image_status():
    return jsonify({"image_translate_requests": translate_image_counter})


# 启动时立即加载
preload_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

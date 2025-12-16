# app.py - Music MV Generator (最終版 - 整合所有功能)

import os
import sys
import uuid
import base64
import json
import shutil
import zipfile
from werkzeug.utils import secure_filename 
from werkzeug.datastructures import FileStorage
# 引入 mock 函式的依賴
import cv2 
from PIL import Image, ImageDraw 
from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS

# --- 導入模組 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 導入所有核心模組
from audio_processor import process_audio_to_lyrics 
from prompt_generation_module import generate_prompts_from_lyrics_text, generate_and_save_prompts_from_srt
import image_generation_module # 包含 SDXL 邏輯
import video_generation_module # 包含 SVD 邏輯
import video_merger_module     # 🌟 新增導入：影片合併模組 (來自 Merge_MV_2.ipynb 邏輯)

# --- 設定 ---
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 設置最大的請求內容長度 (50 MB，為合併 MV 預留更大空間)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

def file_to_base64(path):
    """將檔案讀取為 Base64 字串。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


# --- Mock 函式 (mock_prompt_to_image 僅作為定義保留) ---

def mock_prompt_to_image(prompt, file_id, output_dir):
    """模擬圖片生成，實際寫入一個帶有 Prompt 的 PNG 檔案。"""
    print(f"🖼️ 模擬生成圖片: {prompt[:50]}...")
    img = Image.new('RGB', (512, 512), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), prompt, fill=(255,255,0))
    
    out_path = os.path.join(output_dir, f"{file_id}.png")
    img.save(out_path)
    return out_path


# --- 路由 ---

@app.route("/")
def index():
    return "Backend Server Running. Please access via index.html."


# -------------------------------------------------------------
# Step 1 & 2: 音訊與歌詞處理 
# -------------------------------------------------------------

@app.route("/audio_to_lyrics", methods=["POST"])
def audio_to_lyrics_route():
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "Invalid JSON"}), 400
    
    file_id = str(uuid.uuid4())
    filename = data['filename']
    base_name = data.get('base_name', "Lyrics")
    in_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    
    try:
        b64_str = data['audio_data_url'].split(',')[1] if ',' in data['audio_data_url'] else data['audio_data_url']
        with open(in_path, "wb") as f:
            f.write(base64.b64decode(b64_str))
            
        txt_path, srt_path, error = process_audio_to_lyrics(in_path, OUTPUT_DIR, file_id, base_name=base_name)
        if error: return jsonify({"success": False, "error": error}), 500
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": f"{file_id}_lyrics.txt",
            "base64": file_to_base64(txt_path),
            "base_name": base_name 
        })
    except Exception as e:
        print(f"audio_to_lyrics_route 發生錯誤: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(in_path): os.remove(in_path)


@app.route("/lyrics_to_prompts", methods=["POST"])
def lyrics_to_prompts_route():
    if 'srt_file' in request.files:
        file = request.files['srt_file']
        file_id = str(uuid.uuid4())
        base_name = request.form.get("base_name", "Prompts")
        in_path = os.path.join(UPLOAD_DIR, f"{file_id}_{secure_filename(file.filename)}")
        file.save(in_path)
        
        try:
            success, zip_name, err = generate_and_save_prompts_from_srt(
                in_path, 
                OUTPUT_DIR, 
                file_id, 
                style="Cinematic, Moody, High Resolution", 
                original_filename=base_name
            )
            
            if not success: return jsonify({"success": False, "error": err}), 500
            
            return jsonify({
                "success": True, 
                "mode": "srt_zip",
                "file_id": file_id,
                "zip_name": zip_name,
                "base_name": base_name
            })
            
        except Exception as e:
            print(f"SRT 轉 ZIP 發生錯誤: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
        finally:
            if os.path.exists(in_path): os.remove(in_path)

    else:
        return jsonify({"success": False, "error": "This pipeline requires an SRT file upload."}), 400
        

# -------------------------------------------------------------
# Step 3: Prompt → Image (單圖/批次 核心邏輯)
# -------------------------------------------------------------

@app.route("/prompt_to_image", methods=["POST"])
def prompt_to_image_route():
    mode = request.form.get("mode", "batch")
    
    file_id = str(uuid.uuid4())
    base_name = request.form.get("base_name", "Images")

    if mode == "single":
        # --- 模式 A: 單圖模式 (JSON 文本輸入, PNG 輸出) ---
        json_text = request.form.get("prompt_json_text")
        if not json_text:
            return jsonify({"success": False, "error": "Missing 'prompt_json_text' field (JSON file content)."}), 400

        temp_image_dir = os.path.join(OUTPUT_DIR, f"temp_single_{file_id}")
        os.makedirs(temp_image_dir, exist_ok=True)
            
        try:
            prompt_data = json.loads(json_text)
            
            output_path, error = image_generation_module.generate_image_from_prompt_data(
                prompt_data, 
                temp_image_dir
            )
            
            if error:
                return jsonify({"success": False, "error": f"單圖生成失敗: {error}"}), 500

            b64_image = file_to_base64(output_path)
            
            return jsonify({
                "success": True,
                "base64": b64_image,
                "base_name": secure_filename(base_name) 
            })

        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "JSON 提示詞格式錯誤。"}), 400
        except Exception as e:
            print(f"單圖生成發生錯誤: {e}")
            return jsonify({"success": False, "error": f"單圖生成過程中發生錯誤: {e}"}), 500
        finally:
            if os.path.exists(temp_image_dir): shutil.rmtree(temp_image_dir)


    else: # mode == "batch"
        # --- 模式 B: 批次模式 (ZIP 輸入, ZIP 輸出) ---
        if 'prompt_zip' not in request.files:
            return jsonify({"success": False, "error": "Missing 'prompt_zip' file in batch mode."}), 400
            
        file = request.files['prompt_zip']
        
        in_zip_path = os.path.join(UPLOAD_DIR, f"{file_id}_prompts.zip")
        temp_json_dir = os.path.join(UPLOAD_DIR, f"temp_prompts_{file_id}")
        temp_image_dir = os.path.join(OUTPUT_DIR, f"temp_images_{file_id}")
        
        out_zip_name = f"{file_id}_images.zip"
        out_zip_path = os.path.join(OUTPUT_DIR, out_zip_name)
        
        os.makedirs(temp_json_dir, exist_ok=True)
        os.makedirs(temp_image_dir, exist_ok=True)
        file.save(in_zip_path)

        try:
            with zipfile.ZipFile(in_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_json_dir)

            json_files = [f for f in os.listdir(temp_json_dir) if f.endswith('.json')]
            json_files.sort()
            
            if not json_files:
                return jsonify({"success": False, "error": "ZIP 檔案中未找到任何 .json 提示檔案。"}), 400
                
            print(f"--- 開始 SDXL 圖像生成 (批次)，共 {len(json_files)} 個檔案 ---")
            for i, filename in enumerate(json_files):
                json_path = os.path.join(temp_json_dir, filename)
                
                output_path, error = image_generation_module.generate_image_from_prompt_data(
                    json_path, 
                    temp_image_dir
                )
                
                if error:
                    print(f"❌ 圖像生成錯誤: {error}")
                    
            print("--- SDXL 圖像生成 (批次) 完成 ---")
            
            image_files = [f for f in os.listdir(temp_image_dir) if f.endswith('.png')]
            if not image_files:
                return jsonify({"success": False, "error": "圖像生成失敗或無圖像生成。"}), 500
                
            with zipfile.ZipFile(out_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for filename in image_files:
                    file_path = os.path.join(temp_image_dir, filename)
                    zipf.write(file_path, filename)
                    
            print(f"✅ 圖像 ZIP 檔案成功創建: {out_zip_path}")
            
            return jsonify({
                "success": True,
                "file_id": file_id,
                "zip_name": out_zip_name,
                "base_name": base_name 
            })
            
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 500
        except Exception as e:
            print(f"prompt_to_image_route 發生錯誤: {e}")
            return jsonify({"success": False, "error": f"圖像生成過程中發生未知錯誤: {e}"}), 500
        finally:
            if os.path.exists(in_zip_path): os.remove(in_zip_path)
            if os.path.exists(temp_json_dir): shutil.rmtree(temp_json_dir)
            if os.path.exists(temp_image_dir): shutil.rmtree(temp_image_dir)


# -------------------------------------------------------------
# Step 4: 圖像轉影片路由 (SVD)
# -------------------------------------------------------------

@app.route("/image_to_video", methods=["POST"])
def image_to_video_route():
    if 'image_zip' not in request.files:
        return jsonify({"success": False, "error": "Missing 'image_zip' file."}), 400
        
    file = request.files['image_zip']
    
    file_id = str(uuid.uuid4())
    in_zip_path = os.path.join(UPLOAD_DIR, f"{file_id}_input.zip")
    temp_unzip_dir = os.path.join(UPLOAD_DIR, f"temp_unzip_{file_id}")
    temp_video_dir = os.path.join(OUTPUT_DIR, f"temp_video_{file_id}")
    
    os.makedirs(temp_unzip_dir, exist_ok=True)
    os.makedirs(temp_video_dir, exist_ok=True)
    
    file.save(in_zip_path) 

    image_path = None
    input_base_name = None
    
    try:
        # 1. 解壓縮 ZIP 檔案
        with zipfile.ZipFile(in_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_unzip_dir)

        # 2. 尋找 PNG 圖像 (只取第一個)
        png_files = [f for f in os.listdir(temp_unzip_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            return jsonify({"success": False, "error": "ZIP 檔案中未找到 PNG 圖像。"}), 400
            
        image_filename = png_files[0]
        image_path = os.path.join(temp_unzip_dir, image_filename)
        input_base_name = os.path.splitext(image_filename)[0] 

        # 3. 呼叫 SVD 影片生成
        video_path, error = video_generation_module.generate_video_from_image_path(
            image_path,
            temp_video_dir
        )
        
        if error:
            return jsonify({"success": False, "error": f"影片生成失敗: {error}"}), 500

        # 4. 讀取生成的 MP4 檔案並轉為 Base64 返回
        final_video_path = video_path 
        b64_video = file_to_base64(final_video_path)
        
        return jsonify({
            "success": True, 
            "base64": b64_video, 
            "filename": f"{input_base_name}.mp4"
        })
        
    except Exception as e:
        print(f"image_to_video_route 發生錯誤: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # 清理臨時文件和目錄
        if os.path.exists(in_zip_path): os.remove(in_zip_path)
        if os.path.exists(temp_unzip_dir): shutil.rmtree(temp_unzip_dir)
        if os.path.exists(temp_video_dir): shutil.rmtree(temp_video_dir)
        
        
# -------------------------------------------------------------
# 🌟 Step 5: 影片合併路由 (使用 MoviePy) 🌟
# -------------------------------------------------------------

@app.route("/merge_video_clip", methods=["POST"])
def merge_video_clip_route():
    # 檢查是否上傳了 ZIP 檔案
    if 'mv_zip' not in request.files:
        return jsonify({"success": False, "error": "Missing 'mv_zip' file."}), 400
        
    file = request.files['mv_zip']
    
    file_id = str(uuid.uuid4())
    in_zip_path = os.path.join(UPLOAD_DIR, f"{file_id}_mv_input.zip")
    temp_unzip_dir = os.path.join(UPLOAD_DIR, f"temp_mv_unzip_{file_id}")
    
    os.makedirs(temp_unzip_dir, exist_ok=True)
    
    file.save(in_zip_path) # 儲存上傳的 ZIP 檔案

    final_video_path = None
    
    try:
        print("--- 🎬 Step 5: 開始影片合併 ---")
        # 1. 解壓縮 ZIP 檔案
        with zipfile.ZipFile(in_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_unzip_dir) 

        # 2. 呼叫影片合併模組
        video_path, error = video_merger_module.merge_mv_from_zip_contents(
            temp_unzip_dir,
            OUTPUT_DIR,
            file_id
        )
        
        if error:
            return jsonify({"success": False, "error": f"影片合併失敗: {error}"}), 500

        # 3. 讀取生成的 MP4 檔案並轉為 Base64 返回
        final_video_path = video_path
        base_name = os.path.basename(final_video_path)
        b64_video = file_to_base64(final_video_path)
        
        return jsonify({
            "success": True, 
            "base64": b64_video, 
            "filename": base_name
        })
        
    except Exception as e:
        print(f"merge_video_clip_route 發生錯誤: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # 清理臨時文件和目錄
        if os.path.exists(in_zip_path): os.remove(in_zip_path)
        if os.path.exists(temp_unzip_dir): shutil.rmtree(temp_unzip_dir, ignore_errors=True)
        # 最終影片檔案保留在 OUTPUT_DIR 中
        # if os.path.exists(final_video_path): os.remove(final_video_path)


# -------------------------------------------------------------
# 下載檔案路由 (支援 ZIP, SRT, TXT)
# -------------------------------------------------------------

@app.route("/download/<file_id>/<ftype>", methods=["GET"])
def download_file(file_id, ftype):
    if ftype == 'zip': 
        fname_on_disk_prompt = f"{file_id}_prompts.zip"
        fname_on_disk_image = f"{file_id}_images.zip"

        user_base_name = request.args.get("name", "Downloaded")
        
        target_path = None
        download_name = None
        
        if os.path.exists(os.path.join(OUTPUT_DIR, fname_on_disk_prompt)):
            target_path = fname_on_disk_prompt
            download_name = f"{user_base_name}_prompts.zip"
        elif os.path.exists(os.path.join(OUTPUT_DIR, fname_on_disk_image)):
            target_path = fname_on_disk_image
            download_name = f"{user_base_name}_images.zip"
        else:
            return jsonify({"error": f"File not found on disk for ID: {file_id}. (Prompt/Image ZIP)"}), 404
        
        safe_download_name = secure_filename(download_name)
        
        return send_from_directory(
            OUTPUT_DIR, 
            target_path, 
            as_attachment=True,
            download_name=safe_download_name
        )
        
    elif ftype in ['srt', 'txt']:
        fname_on_disk = f"{file_id}_lyrics.{ftype}"
        user_base_name = request.args.get("name", "Lyrics")
        download_name = f"{user_base_name}.{ftype}"
        safe_download_name = secure_filename(download_name)
        
        if not os.path.exists(os.path.join(OUTPUT_DIR, fname_on_disk)):
            return jsonify({"error": f"File not found on disk: {fname_on_disk}"}), 404
        
        return send_from_directory(
            OUTPUT_DIR, 
            fname_on_disk, 
            as_attachment=True, 
            download_name=safe_download_name
        )
    
    return jsonify({"error": "Invalid file type."}), 400


# -------------------------------------------------------------
# 啟動應用程式
# -------------------------------------------------------------

if __name__ == "__main__":
    # 創建一個空的 mock_image.png (用於舊的 mock 函式)
    mock_img_path = os.path.join(BASE_DIR, "mock_image.png")
    if not os.path.exists(mock_img_path):
        try:
            img = Image.new('RGB', (640, 480), color = 'red')
            d = ImageDraw.Draw(img)
            d.text((10, 10), "MOCK IMAGE", fill=(255, 255, 255))
            img.save(mock_img_path)
        except Exception:
            pass 

    print("-" * 50)
    print("🚀 正在啟動後端服務...")
    
    # 🌟 啟動時預載入所有模型
    try:
        print("--- 預載入 SDXL T2I 模型 ---")
        image_generation_module.initialize_sdxl()
    except Exception as e:
        print(f"🚨 SDXL 模型初始化失敗: {e}")
        
    try:
        print("--- 預載入 SVD I2V 模型 ---")
        video_generation_module.initialize_svd()
    except Exception as e:
        print(f"🚨 SVD 模型初始化失敗: {e}")
        
    print("-" * 50)
    print(f"📂 上傳目錄: {UPLOAD_DIR}")
    print(f"📦 輸出目錄: {OUTPUT_DIR}")
    print(f"🌍 服務運行於 http://0.0.0.0:5000")
    print("-" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
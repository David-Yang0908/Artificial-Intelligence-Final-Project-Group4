# image_generation_module.py - 支援 SDXL T2I 載入、單圖及批次生成

import os
import torch
import json
import gc
import uuid
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import snapshot_download

# --- 核心設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 保持與 video_v2.ipynb 和 app.py 一致的模型路徑配置
BASE_MODEL_CACHE_PATH = os.path.join(BASE_DIR, "local_models") 
SDXL_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_MODEL_PATH = os.path.join(BASE_MODEL_CACHE_PATH, "sdxl_base")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE_T2I = None # 單例模型變數

def check_and_download_sdxl_model():
    """檢查 SDXL 模型是否存在，不存在則下載 (不重複下載)。"""
    if os.path.exists(SDXL_MODEL_PATH) and os.listdir(SDXL_MODEL_PATH):
        print(f"✅ SDXL 模型已存在: {SDXL_MODEL_PATH}")
        return True
    
    print(f"--- 偵測到 SDXL 模型不存在: {SDXL_MODEL_PATH}，正在從 Hugging Face 下載 {SDXL_REPO_ID} ---")
    os.makedirs(SDXL_MODEL_PATH, exist_ok=True)
    try:
        # 使用 snapshot_download 下載模型到本地路徑，如果已存在則不會重複下載
        snapshot_download(
            repo_id=SDXL_REPO_ID, 
            local_dir=SDXL_MODEL_PATH, 
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt", "*.bin"] 
        )
        print(f"✅ {SDXL_REPO_ID} 模型下載完成。")
        return True
    except Exception as e:
        print(f"❌ 下載 {SDXL_REPO_ID} 失敗: {e}")
        return False


def load_sdxl_pipeline():
    """載入 SDXL Pipeline (單例模式)"""
    global PIPE_T2I
    if PIPE_T2I is not None:
        return PIPE_T2I # 模型已載入，直接返回

    if not check_and_download_sdxl_model():
        raise RuntimeError("SDXL 模型下載或檢查失敗，無法載入 Pipeline。")

    print("\n--- 正在載入 Stable Diffusion XL (T2I) 模型到記憶體 ---")
    try:
        # 使用與 Notebook 相同的配置
        pipe_t2i = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True,
        ).to(DEVICE)

        if DEVICE == "cuda":
            # 啟用 CPU Offload 以節省 VRAM
            pipe_t2i.enable_model_cpu_offload() 
        
        PIPE_T2I = pipe_t2i # 設置單例變數
        print("✅ Stable Diffusion XL Pipeline 載入完成。")
        return PIPE_T2I
    except Exception as e:
        print(f"❌ 載入 SDXL 失敗: {e}")
        cleanup_sdxl_memory() 
        raise RuntimeError(f"SDXL 模型載入失敗: {e}")

def initialize_sdxl():
    """初始化模型，在 Flask 啟動前調用"""
    try:
        load_sdxl_pipeline()
    except Exception as e:
        # 僅發出警告，讓服務繼續運行
        print(f"⚠️ 警告: SDXL 預載入失敗: {e}") 
        pass


def cleanup_sdxl_memory():
    """清理 SDXL 模型記憶體。"""
    global PIPE_T2I
    if PIPE_T2I is not None:
        print("\n--- 正在釋放 SDXL 模型記憶體 ---")
        del PIPE_T2I
        PIPE_T2I = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ 記憶體已清理。")


def generate_image_from_prompt_data(prompt_source: dict | str, image_output_dir: str):
    """
    根據 JSON Prompt 數據 (dict) 或 JSON 檔案路徑 (str) 生成圖像，並儲存為 PNG 檔案。
    
    Args:
        prompt_source: 包含 Prompt 資訊的字典 (單圖模式) 或 JSON 檔案路徑 (批次模式)。
        image_output_dir: 圖像儲存目錄。
        
    Returns:
        (str, None) or (None, str): 成功時返回 (輸出路徑, None)，失敗時返回 (None, 錯誤訊息)。
    """
    try:
        pipe = load_sdxl_pipeline()
    except Exception as e:
        return None, str(e)
        
    # --- 處理輸入數據 ---
    if isinstance(prompt_source, dict):
        # 單圖模式：直接使用字典數據
        prompt_data = prompt_source
        base_name = str(uuid.uuid4()) # 單圖模式使用 UUID 作為檔名基礎
        is_single_mode = True
        
    elif isinstance(prompt_source, str) and os.path.exists(prompt_source):
        # 批次模式：讀取 JSON 檔案
        json_path = prompt_source
        base_name = os.path.basename(json_path).replace('.json', '')
        is_single_mode = False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
        except Exception as e:
            return None, f"無法讀取或解析 JSON 檔案 {base_name}.json: {e}"
    else:
        return None, "無效的 Prompt 輸入來源 (非字典或有效檔案路徑)"

    image_output_path = os.path.join(image_output_dir, f"{base_name}.png")

    # 批次模式才進行跳過檢查，單圖模式總是會生成新的 UUID 檔案
    if not is_single_mode and os.path.exists(image_output_path):
        print(f"  -> ⏩ 跳過: {base_name}.png (圖像已存在)")
        return image_output_path, None

    try:
        # 驗證必要的字段
        if 'shot_description' not in prompt_data or 'style_keywords' not in prompt_data:
             raise ValueError("JSON 格式錯誤：缺少 'shot_description' 或 'style_keywords' 字段。")
             
        # 組合 T2I Prompt (根據 video_v2.ipynb 邏輯)
        prompt = f"{prompt_data['shot_description']}, {prompt_data['style_keywords']}"
        negative_prompt = prompt_data.get('negative_prompt', "blurry, worst quality, noise, bad anatomy, deformed, text, watermark")
        
        print(f"  -> 🖼️ 正在生成圖像: {base_name}.png (Prompt: {prompt[:50]}...)")

        # 執行圖像生成 (SDXL)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

        # 儲存到輸出目錄
        image.save(image_output_path)
        return image_output_path, None

    except ValueError as e:
        return None, f"數據錯誤 ({base_name}): {e}"
    except Exception as e:
        # 清理記憶體以防萬一
        cleanup_sdxl_memory() 
        return None, f"圖像生成失敗 ({base_name}): {e}"

# 確保在匯入時可以檢查模型存在性 (如果需要)
# check_and_download_sdxl_model()
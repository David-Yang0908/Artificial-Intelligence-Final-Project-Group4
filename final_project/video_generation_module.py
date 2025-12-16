# video_generation_module.py

import os
import gc
import torch
import uuid
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# --- 核心設定 (基於 video_v2.ipynb) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_CACHE_PATH = os.path.join(BASE_DIR, "local_models") 
SVD_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
SVD_MODEL_PATH = os.path.join(BASE_MODEL_CACHE_PATH, "svd_base")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE_I2V = None # 單例模型變數

# SVD 運行參數 (來自 notebook)
SVD_NUM_FRAMES = 25
SVD_DECODE_CHUNK_SIZE = 4
SVD_FPS = 6 

def flush_memory():
    """清理 CUDA 記憶體並運行 Python 垃圾回收 (來自 notebook)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # print("✅ 記憶體已清理。")

def check_and_download_svd_model():
    """檢查 SVD 模型是否存在，不存在則下載 (不重複下載)。"""
    if os.path.exists(SVD_MODEL_PATH) and os.listdir(SVD_MODEL_PATH):
        print(f"✅ SVD 模型已存在: {SVD_MODEL_PATH}")
        return True
    
    print(f"--- 偵測到 SVD 模型不存在: {SVD_MODEL_PATH}，正在從 Hugging Face 下載 {SVD_REPO_ID} ---")
    os.makedirs(SVD_MODEL_PATH, exist_ok=True)
    try:
        snapshot_download(
            repo_id=SVD_REPO_ID, 
            local_dir=SVD_MODEL_PATH, 
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt"]
        )
        print(f"✅ {SVD_REPO_ID} 模型下載完成。")
        return True
    except Exception as e:
        print(f"❌ 下載 {SVD_REPO_ID} 失敗: {e}")
        return False


def load_svd_pipeline():
    """載入 SVD Pipeline (單例模式)"""
    global PIPE_I2V
    if PIPE_I2V is not None:
        return PIPE_I2V

    if not check_and_download_svd_model():
        raise RuntimeError("SVD 模型下載或檢查失敗，無法載入 Pipeline。")

    print("\n--- 正在載入 Stable Video Diffusion (I2V) 模型到記憶體 ---")
    try:
        pipe_i2v = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)

        if DEVICE == "cuda":
            # SVD 記憶體優化設置 (來自 notebook)
            pipe_i2v.enable_model_cpu_offload()
            pipe_i2v.unet.enable_forward_chunking()
        
        PIPE_I2V = pipe_i2v
        print("✅ Stable Video Diffusion Pipeline 載入完成。")
        return PIPE_I2V
    except Exception as e:
        print(f"❌ 載入 SVD 失敗: {e}")
        cleanup_svd_memory()
        raise RuntimeError(f"SVD 模型載入失敗: {e}")

def initialize_svd():
    """初始化 SVD 模型，在 Flask 啟動前調用"""
    try:
        load_svd_pipeline()
    except Exception as e:
        print(f"⚠️ 警告: SVD 預載入失敗: {e}")
        pass

def cleanup_svd_memory():
    """清理 SVD 模型記憶體。"""
    global PIPE_I2V
    if PIPE_I2V is not None:
        print("\n--- 正在釋放 SVD 模型記憶體 ---")
        del PIPE_I2V
        PIPE_I2V = None
        flush_memory()
        print("✅ SVD 記憶體已清理。")


def generate_video_from_image_path(image_path: str, output_dir: str):
    """
    從單張圖像路徑生成影片片段。
    
    Args:
        image_path: 輸入圖像的檔案路徑。
        output_dir: 影片儲存目錄。
        
    Returns:
        (str, None) or (None, str): 成功時返回 (輸出路徑, None)，失敗時返回 (None, 錯誤訊息)。
    """
    try:
        pipe = load_svd_pipeline()
    except Exception as e:
        return None, str(e)
        
    base_name = os.path.basename(image_path).replace('.png', '')
    video_output_path = os.path.join(output_dir, f"{base_name}.mp4")

    try:
        # 讀取圖像
        init_image = Image.open(image_path).convert("RGB")
        # 調整圖像大小以符合 SVD 要求 (來自 notebook: 1024x576)
        init_image = init_image.resize((1024, 576))

        print(f"🎥 正在生成影片: {base_name}.mp4")

        # 執行影片生成 (SVD)
        # 設置種子以確保結果可重現 (來自 notebook)
        generator = torch.Generator(device=DEVICE).manual_seed(42)

        video_frames = pipe(
            init_image,
            decode_chunk_size=SVD_DECODE_CHUNK_SIZE,
            num_frames=SVD_NUM_FRAMES,
            generator=generator,
        ).frames[0]

        # 儲存為影片檔案 (.mp4) (來自 notebook)
        export_to_video(video_frames, video_output_path, fps=SVD_FPS)
        
        return video_output_path, None

    except Exception as e:
        cleanup_svd_memory()
        return None, f"影片生成失敗 ({base_name}): {e}"
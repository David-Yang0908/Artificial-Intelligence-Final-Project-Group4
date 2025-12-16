# audio_processor.py - 修正版
import os
import subprocess
import whisper
from whisper.utils import get_writer
import torch
import shutil
import time # 引入 time

# --- 全域模型載入區域 (只在 Flask 服務啟動時載入一次) ---
WHISPER_MODEL_SIZE = "medium.en"  # 您可以根據需求調整模型大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("-" * 50)
print(f"🤖 正在預載入 Whisper 模型 ({WHISPER_MODEL_SIZE}) 到 {DEVICE.upper()}...")
print("此步驟可能需要 30 秒至數分鐘，請耐心等待...")
try:
    start_time = time.time()
    # 這裡會發生一次性的長時間阻塞，但這是必要的
    GLOBAL_WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
    end_time = time.time()
    print(f"✅ Whisper 模型載入完成！耗時: {end_time - start_time:.2f} 秒")
except Exception as e:
    print(f"❌ Whisper 載入失敗: {e}")
    GLOBAL_WHISPER_MODEL = None
    # 載入失敗時，Flask 服務可能會立即崩潰，需要檢查環境依賴是否齊全。
print("-" * 50)
# --- 結束全域模型載入區域 ---


def separate_vocals(input_audio_path, output_dir):
    """使用 Demucs 分離音軌，僅保留人聲。"""
    print(f"分離音軌: {input_audio_path}")
    
    # Demucs 預設會創建一個 'htdemucs' 資料夾在 output_dir 內
    command = [
        "demucs",
        "--two-stems=vocals",
        "-o", output_dir,
        input_audio_path
    ]
    
    try:
        # 使用 check=True 確保指令失敗時拋出異常
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("音軌分離完成")

        filename = os.path.splitext(os.path.basename(input_audio_path))[0]
        # Demucs 輸出路徑格式: output_dir/htdemucs/input_filename/vocals.wav
        vocals_path = os.path.join(output_dir, "htdemucs", filename, "vocals.wav")

        if os.path.exists(vocals_path):
            return vocals_path, None
        else:
            # 嘗試讀取 Demucs 的錯誤輸出
            error_output = "Demucs completed but output file not found. Check if FFmpeg is working."
            raise FileNotFoundError(f"找不到分離後的檔案: {vocals_path}. {error_output}")

    except subprocess.CalledProcessError as e:
        error_message = f"Demucs 執行錯誤. Stderr: {e.stderr.decode()}"
        print(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message
        
    return None, "未知錯誤"


def transcribe_and_export(audio_path, output_dir, language="en"):
    """使用 全域載入的 Whisper 模型 轉錄音訊為 SRT 和 TXT 檔案。"""
    if GLOBAL_WHISPER_MODEL is None:
        # 應在啟動時就發現這個錯誤
        return None, None, "Whisper 模型未成功載入，無法轉錄。"

    print(f"轉錄歌詞 (模型: {WHISPER_MODEL_SIZE}, 語言: {language})")

    try:
        # 🚨 使用已經載入的全域模型，避免重複載入
        model = GLOBAL_WHISPER_MODEL 

        # 1. 轉錄音訊
        result = model.transcribe(
            audio_path, 
            language=language,
            # 可以根據音樂語言調整這個 initial_prompt
            initial_prompt="the audio is a song, please transcribe the lyrics accurately."
        )

        # 2. 輸出到指定的資料夾
        srt_writer = get_writer("srt", output_dir)
        srt_writer(result, audio_path)

        txt_writer = get_writer("txt", output_dir)
        txt_writer(result, audio_path)
        
        # 輸出檔案名稱會是音檔的 basename (不含副檔名) + .txt/.srt，這裡固定為 'vocals'
        output_name_base = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 返回輸出檔案的路徑
        return os.path.join(output_dir, f"{output_name_base}.txt"), os.path.join(output_dir, f"{output_name_base}.srt"), None

    except Exception as e:
        error_message = f"Whisper 轉錄失敗: {e}"
        print(error_message)
        return None, None, error_message


def process_audio_to_lyrics(input_audio_path, output_dir, file_id, base_name="lyrics"): # 👈 記得這個函式現在需要接收 base_name
    """
    主處理流程: 分離人聲 -> 轉錄歌詞 -> 回傳結果路徑。
    """
    
    # 確保輸出資料夾存在
    temp_demucs_dir = os.path.join(output_dir, f"temp_demucs_{file_id}")
    os.makedirs(temp_demucs_dir, exist_ok=True)
    
    # 1. 分離人聲 (輸出到 temp_demucs_dir)
    vocals_file, demucs_error = separate_vocals(input_audio_path, temp_demucs_dir)
    
    if demucs_error:
        # 清理暫存資料夾
        shutil.rmtree(temp_demucs_dir, ignore_errors=True)
        return None, None, demucs_error

    # 2. 轉錄歌詞 (輸出到 output_dir)
    txt_path, srt_path, whisper_error = transcribe_and_export(
        vocals_file, 
        output_dir=output_dir, 
        language="en"
    )
    
    # 清理 Demucs 暫存資料夾
    shutil.rmtree(temp_demucs_dir, ignore_errors=True)
    
    if whisper_error:
        return None, None, whisper_error
        
    # 3. 將生成的檔案重新命名為我們想要的格式 (使用 file_id)
    # 檔案命名格式仍然是 {file_id}_lyrics.{ftype}，以確保 app.py 的 download 路由能找到它。
    final_txt_path = os.path.join(output_dir, f"{file_id}_lyrics.txt")
    final_srt_path = os.path.join(output_dir, f"{file_id}_lyrics.srt")
    
    try:
        os.rename(txt_path, final_txt_path)
        os.rename(srt_path, final_srt_path)
    except FileNotFoundError as e:
        # 由於 Demucs 輸出名稱是 'vocals.wav'，Whisper 輸出的是 'vocals.txt' 和 'vocals.srt'
        # 如果找不到，檢查是不是舊版 Demucs 的問題，但這裡的邏輯是正確的。
        return None, None, f"重新命名轉錄結果時出錯 (找不到 {txt_path} 或 {srt_path}): {e}"
        
    # 回傳 TXT 檔案路徑
    return final_txt_path, final_srt_path, None
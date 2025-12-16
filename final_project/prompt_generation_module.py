# prompt_generation_module.py - 最終修正版 (JSON 檔名顯示開始和結束時間，並優化連續鏡頭連貫性)

import os
import sys
import json
import re
import time
import uuid
import zipfile
import shutil
from datetime import timedelta
import requests
from typing import Union # 🌟 修正: 引入 Union 供類型註記使用

# 🚨 引入 Groq SDK
try:
    from groq import Groq, RateLimitError, APIError 
    from groq.types.chat import ChatCompletionMessageParam
except ImportError:
    print("❌ 錯誤: Groq SDK (pip install groq) 未安裝。")
    sys.exit(1)

# --- 核心設定 ---
GROQ_MODEL_NAME = 'llama-3.1-8b-instant'
MAX_FIELD_TOKENS = 70  # 設置為 44 Token 限制
MAX_CLIP_DURATION = 3.0  # 設置每 3.0 秒一個分鏡

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CLIENT = None
GROQ_CLIENT_READY = False

if GROQ_API_KEY:
    try:
        GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
        GROQ_CLIENT_READY = True
        print(f"✅ Groq API Client 準備就緒 ({GROQ_MODEL_NAME})。")
    except Exception as e:
        print(f"❌ Groq 客戶端初始化失敗: {e}")
else:
    print("❌ 警告: 環境變數 GROQ_API_KEY 未設定。所有 Prompt 生成將失敗。")


# --- 輔助函數 (時間處理與 SRT 解析) ---

def srt_time_to_seconds(srt_time_str):
    """將 SRT 時間格式轉換為秒數"""
    parts = re.split(r'[:|;|,|\.]', srt_time_str)
    try:
        if len(parts) >= 4:
            h, m, s, ms = map(int, parts[:4])
            return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms).total_seconds()
        return 0.0
    except ValueError:
        return 0.0

def parse_srt(file_path):
    """解析 SRT 檔案並返回歌詞片段列表"""
    segments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception:
        return []

    blocks = re.split(r'\n\n+', content)
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3 and lines[0].isdigit():
            timecode = lines[1].strip()
            lyric_text = ' '.join(lines[2:]).strip()
            if '-->' in timecode:
                start_str, end_str = [t.strip() for t in timecode.split('-->')]
                segments.append({
                    'start_time_sec': srt_time_to_seconds(start_str),
                    'end_time_sec': srt_time_to_seconds(end_str),
                    'lyric': lyric_text
                })
    return segments

def format_time_for_filename(td):
    """將 timedelta 格式化為檔案名稱用的時間字串 (00h00m00s000ms)"""
    total_seconds = int(td.total_seconds())
    ms = int((td.total_seconds() - total_seconds) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}ms"

# 備用方案 (修正: 接受 shot_count 而非 lyric_lines)
def create_single_shot(num, description, style):
    return {
        "shot_description": f"{description[:40] if description else 'establishing shot'}, {style}",
        "style_keywords": "cinematic, photorealistic, high quality, vibrant",
        "negative_prompt": "blurry, low quality, abstract, cartoon, deformed, text, watermark" 
    }

def create_fallback(shot_count: int, style: str):
    """創建指定數量的通用備用分鏡提示詞"""
    print(f"創建 {shot_count} 個備用分鏡")
    shots = []
    for i in range(shot_count):
        description = f"Shot {i+1} establishing the mood of the music video, ambient lighting, subtle camera movement"
        shots.append(create_single_shot(i + 1, description, style))
    return shots


# --- Groq API 呼叫的核心內部函式 ---

def _call_groq_api(system_instruction, user_prompt, shot_count, json_mode=False):
    """執行對 Groq API 的單次呼叫，包含 3 次重試機制。"""
    if not GROQ_CLIENT_READY:
        raise ConnectionError("Groq API Client is not initialized.")
        
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"呼叫 Groq API ({GROQ_MODEL_NAME}) - 第 {attempt + 1} 次嘗試...")
            
            response = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                # 調整 max_tokens 以適應更多的提示詞數量
                max_tokens=min(8192, shot_count * 250 + 512), 
                response_format={"type": "json_object"} if json_mode else None
            )
            
            generated_text = response.choices[0].message.content.strip()
            print(f"✅ 成功! Groq 輸出長度: {len(generated_text)} 字符")
            return generated_text
            
        except (RateLimitError, APIError) as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** (attempt + 1) 
                print(f"   ❌ Groq API 錯誤: {e}. 於 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                raise ConnectionError(f"Grok API 連續 {MAX_RETRIES} 次呼叫失敗: {e}")
        
        except Exception as e:
            raise ConnectionError(f"未預期錯誤發生: {e}")

# --- 核心邏輯：功能 A (純文字) ---
def generate_prompts_from_lyrics_text(lyrics_path, output_prompt_path):
    """舊有的簡單邏輯：輸入純文字歌詞 -> 輸出純文字 Prompts"""
    if not GROQ_CLIENT_READY: return {"success": False, "error": "Groq API Client Error"}
    try:
        with open(lyrics_path, 'r', encoding='utf-8') as f: lyrics_text = f.read()
        
        system_instruction = f"""You are a creative AI image prompt designer. Generate 10 distinct, cinematic AI image prompts based on the user's lyrics.
CRITICAL RULE: Each prompt MUST be under {MAX_FIELD_TOKENS} words. Output strictly one prompt per line, with NO introduction."""
        
        user_prompt = f"Lyrics:\n{lyrics_text}"
        
        generated_text = _call_groq_api(system_instruction, user_prompt, shot_count=10, json_mode=False)
        
        with open(output_prompt_path, 'w', encoding='utf-8') as f:
            f.write(generated_text)
            
        generated_lines = len([line.strip() for line in generated_text.split('\n') if line.strip()])
            
        return {"success": True, "output": f"Generated {generated_lines} prompts from text."}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- 核心邏輯：功能 B (SRT 進階生成) ---
def generate_and_save_prompts_from_srt(srt_path, output_dir, file_id, style="Cinematic, Moody, High Resolution", original_filename="prompts"):
    """進階邏輯：輸入 SRT -> 分割時間軸 -> 輸出 ZIP"""
    if not GROQ_CLIENT_READY: return False, None, "Groq API Client Error"
    
    # 1. 解析 SRT 並進行時間軸細分 (Sub-shot Creation)
    segments = parse_srt(srt_path)
    if not segments: return False, None, "SRT 檔案解析失敗或內容為空。"
    
    sub_shots = [] # 新的細分鏡頭列表
    for i, seg in enumerate(segments):
        start_sec = seg['start_time_sec']
        end_sec = seg['end_time_sec']
        duration = end_sec - start_sec
        
        # 決定需要多少子鏡頭
        num_sub_clips = max(1, int(duration / MAX_CLIP_DURATION))
        # 如果不是整數倍，再加一個片段
        if duration % MAX_CLIP_DURATION != 0.0 and duration > 0.0:
             num_sub_clips += 1
        
        base_duration = duration / num_sub_clips
        current_start_sec = start_sec

        for j in range(num_sub_clips):
            sub_end_sec = current_start_sec + base_duration
            if j == num_sub_clips - 1:
                sub_end_sec = end_sec # 確保最後一個片段結束時間精確對齊

            sub_shots.append({
                'shot_index': f"{i+1}-{j+1}", # 原始編號-子編號
                'lyric_line': seg['lyric'],
                'start_time_sec': current_start_sec,
                'end_time_sec': sub_end_sec,
                'is_first_sub_shot': (j == 0),
                'is_last_sub_shot': (j == num_sub_clips - 1),
                'total_sub_shots': num_sub_clips
            })
            current_start_sec = sub_end_sec

    # 準備傳遞給 Groq 的輸入文本
    shot_count = len(sub_shots)
    input_text_for_groq = ""
    for shot in sub_shots:
        if shot['is_first_sub_shot']:
            # 僅在每個新的歌詞行開始時顯示歌詞內容
            input_text_for_groq += f"\n--- LYRIC SHOT {shot['shot_index'].split('-')[0]} ---\n"
            input_text_for_groq += f"Lyric: {shot['lyric_line']}\n"
        
        # 對於連續鏡頭，標註是第幾個子鏡頭
        input_text_for_groq += f"[{shot['shot_index']}] Duration: {shot['end_time_sec'] - shot['start_time_sec']:.2f}s.\n"

    
    # 2. 呼叫 Groq (要求 JSON 格式)
    system_instruction = f"""You are a master music video director creating a highly cinematic sequence. Your task is to generate EXACTLY {shot_count} unique image prompts (one for each time slot provided below).
**CRITICAL RULES:**
1. **Output Format:** MUST be a valid JSON array of {shot_count} objects. Format: [{{{{ "shot_index": "...", "shot_description": "...", "style_keywords": "...", "negative_prompt": "..." }}}}, ...] (Include ALL four keys).
2. **Coherence & Storyline:** Establish an overarching story/mood based on the lyrics. Consecutive shots belonging to the same LYRIC SHOT (e.g., [1-1], [1-2], [1-3]) MUST describe a visually **continuous or progressive action** (e.g., a zoom in, a slow pan, a subtle change in pose, or movement towards a goal). DO NOT repeat the exact same 'shot_description' for consecutive sub-shots.
3. **Prompt Constraints:** The 'shot_description', 'style_keywords', and 'negative_prompt' fields MUST each be under {MAX_FIELD_TOKENS} tokens.
4. **Style:** MV STYLE: {style}.
5. **Safety:** 'negative_prompt' MUST include 'blurry, worst quality, abstract, cartoon, deformed, text, watermark'.
6. **Integrity:** Ensure the 'shot_index' value EXACTLY matches the input index (e.g., [1-1]).
Start with [ and end with ]. NO markdown, NO explanations."""

    user_prompt = f"""Generate {shot_count} continuous image prompts.
MV STYLE: {style}
LYRICS & SHOT INDEXES ({shot_count} total shots):\n{input_text_for_groq}
Output JSON array of {shot_count} shots. Start with ["""
    
    prompts_data = []
    try:
        generated_text = _call_groq_api(system_instruction, user_prompt, shot_count, json_mode=True)
        
        # 3. JSON 解析與修復
        text = generated_text
        if "```json" in text.lower(): text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE).strip()
        if text.endswith("```"): text = text[:-3]
        
        prompts_data = json.loads(text)
        if isinstance(prompts_data, dict): prompts_data = [prompts_data]
        
    except ConnectionError as e:
        return False, None, str(e) 
    except Exception as e:
        # 🌟 修正: 不再依賴未定義的 lyric_lines，使用 shot_count
        print(f"Groq JSON 解析失敗，使用備用提示詞: {e}") 
        prompts_data = create_fallback(shot_count, style) 

    # 4. 儲存單個 JSON 檔案 (使用原始檔名)
    temp_dir = os.path.join(output_dir, f"temp_{file_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 確保 prompts_data 數量和 sub_shots 數量一致
    if len(prompts_data) != shot_count:
        print(f"❌ 警告: Groq 輸出的提示詞數量 {len(prompts_data)} 與目標數量 {shot_count} 不符。正在嘗試使用備用提示詞。")
        # 再次嘗試備用方案
        prompts_data = create_fallback(shot_count, style)
    
    # 創建一個映射表 (字典)
    prompt_map = {item.get('shot_index'): item for item in prompts_data if 'shot_index' in item}
    
    
    for i, shot in enumerate(sub_shots):
        shot_index = shot['shot_index']
        p_data = prompt_map.get(shot_index)
        
        # 如果找不到對應的 shot_index，使用一個基礎的提示詞
        if not p_data:
            print(f"❌ 警告: 找不到 shot_index {shot_index} 的提示詞，使用基本提示詞。")
            p_data = create_single_shot(
                i + 1, 
                shot['lyric_line'], # 使用歌詞行作為描述基礎
                style
            )
        
        start_td = timedelta(seconds=shot['start_time_sec'])
        end_td = timedelta(seconds=shot['end_time_sec']) 
        
        start_time_fn = format_time_for_filename(start_td)
        end_time_fn = format_time_for_filename(end_td)
        
        # 🚨 關鍵修正：JSON 檔名使用 {原始檔名}_{開始時間}_to_{結束時間}.json
        output_filename = f"{original_filename}_{start_time_fn}_to_{end_time_fn}.json"
        output_filepath = os.path.join(temp_dir, output_filename)
        
        # 確保 JSON 格式符合 Step 3 需求的精簡格式
        output_json = {
            "shot_description": p_data.get('shot_description', f"Shot {shot_index} for: {shot['lyric_line'][:40]}"),
            "style_keywords": p_data.get('style_keywords', "cinematic, moody, high resolution"),
            "negative_prompt": p_data.get('negative_prompt', "blurry, worst quality, abstract, cartoon, deformed, text, watermark")
        }
        
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # 寫入時不再包含 'shot_index'，以保持 JSON 內容的簡潔性
            json.dump(output_json, outfile, indent=2, ensure_ascii=False)

    # 5. 打包 ZIP
    zip_name = f"{file_id}_prompts.zip"
    zip_path = os.path.join(output_dir, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # 僅將檔案名稱（不包含 temp_file_id/）寫入 ZIP
                zf.write(file_path, os.path.basename(file_path)) 
                
    shutil.rmtree(temp_dir) 

    return True, zip_name, None
# video_merger_module.py (最終修剪版 - 無字幕功能)

import os
import re
import shutil
import zipfile

# -------------------------------------------------------------
# 🌟 移除 TextClip, SubtitlesClip 相關導入
# -------------------------------------------------------------
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    # 移除 TextClip, CompositeVideoClip
    concatenate_videoclips,
    vfx,
)
# 移除 from moviepy.video.tools.subtitles import SubtitlesClip
# 移除 try...from moviepy.config import change_settings...

# -----------------------------------------------
# 輔助函式
# -----------------------------------------------

def parse_new_time_format(h, m, s, ms):
    """將 h, m, s, ms 轉換為總秒數 (float)"""
    total_seconds = (int(h) * 3600) + (int(m) * 60) + int(s) + (int(ms) / 1000.0)
    return total_seconds

def get_start_time_from_path(file_path):
    """從檔案路徑解析影片片段的開始時間，用於排序。"""
    filename = os.path.basename(file_path)
    pattern = r'_(\d+)h(\d+)m(\d+)s(\d+)ms_to_'
    match = re.search(pattern, filename)
    if match:
        return parse_new_time_format(match.group(1), match.group(2), match.group(3), match.group(4))
    return 999999.0

# -----------------------------------------------
# 核心合併邏輯
# -----------------------------------------------

def merge_mv_from_zip_contents(temp_unzip_dir: str, output_dir: str, file_id: str):
    """
    從解壓縮後的資料夾中讀取片段和音訊，進行合併 (不包含字幕)。
    """
    
    video_files = []
    mp3_files = []
    
    print(f"   正在掃描 ZIP 內容: {temp_unzip_dir}")
    for root, dirs, files in os.walk(temp_unzip_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.lower().endswith('.mp4'):
                video_files.append(full_path)
            elif f.lower().endswith('.mp3'):
                mp3_files.append(full_path)
            
    if not video_files:
        return None, "ZIP 檔案中未找到任何 MP4 影片片段 (請確認副檔名為 .mp4)。"

    video_files.sort(key=get_start_time_from_path)
    
    clips = []
    print(f"   預計處理 {len(video_files)} 個片段...")

    for file_path in video_files:
        video_filename = os.path.basename(file_path)
        pattern = r'_(\d+)h(\d+)m(\d+)s(\d+)ms_to_(\d+)h(\d+)m(\d+)s(\d+)ms'
        match = re.search(pattern, video_filename)

        if match:
            start_time = parse_new_time_format(match.group(1), match.group(2), match.group(3), match.group(4))
            end_time = parse_new_time_format(match.group(5), match.group(6), match.group(7), match.group(8))
            target_duration = end_time - start_time

            try:
                clip = VideoFileClip(file_path)
                if abs(clip.duration - target_duration) > 0.01:
                    clip = clip.fx(vfx.speedx, final_duration=target_duration)
                clips.append(clip)
            except Exception as e:
                print(f"⚠️ 無法讀取片段 {video_filename}: {e}")
        else:
            print(f"⚠️ 檔名格式不符，跳過: {video_filename}")

    if not clips:
        return None, "沒有成功載入任何有效的影片片段。"

    print(f"   ✅ 成功載入 {len(clips)} 個片段，開始串接...")
    try:
        # 🌟 影片串接
        final_video = concatenate_videoclips(clips, method="compose")
    except Exception as e:
         return None, f"影片串接失敗: {e}"

    # 🌟 2. 移除字幕處理區塊

    # 3. 合成背景音樂 (.mp3)
    final_video_with_audio = final_video
    
    if mp3_files:
        audio_path = mp3_files[0]
        print(f"   正在合成背景音樂 ({os.path.basename(audio_path)})...")
        try:
            audio = AudioFileClip(audio_path)
            # 確保影片長度與音訊一致
            final_duration = min(final_video_with_audio.duration, audio.duration)
            final_video_with_audio = final_video_with_audio.set_audio(audio).set_duration(final_duration)
        except Exception as e:
            print(f"❌ 錯誤：無法讀取音樂檔: {e}")
    else:
        print("   未找到 MP3 音樂檔，將輸出無音訊影片。")
        
    
    # 4. 寫入最終影片
    output_filename = f"{file_id}_full_mv.mp4"
    output_filepath = os.path.join(output_dir, output_filename)
    
    print(f"   開始輸出影片... (儲存至: {output_filepath})")
    
    try:
        final_video_with_audio.write_videofile(
            output_filepath, 
            fps=24, 
            codec='libx264', 
            audio_codec='aac', 
            preset='faster',
            verbose=False,
            logger=None
        )
        print("   ✅ 完整 MV 製作完成！")
        return output_filepath, None
    except Exception as e:
        return None, f"影片寫入失敗: {e}"
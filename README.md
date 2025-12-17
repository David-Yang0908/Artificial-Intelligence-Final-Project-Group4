🎵 Music MV Generator (Hybrid Mode)
本專案是一個整合式的 AI 音樂影片生成系統。使用者只需上傳音訊檔案，系統即可自動完成從人聲分離、歌詞轉錄、影像提示詞 (Prompt) 生成、AI 圖像生成到最終影片合併的全自動化流程。

### demo 影片連結:  https://www.youtube.com/watch?v=-M3SLxqKcZs
### github 連結:    https://github.com/David-Yang0908/Artificial-Intelligence-Final-Project-Group4.git
📖 專案簡介
本系統採用「混合模式 (Hybrid Mode)」，核心功能分為五大步驟：

Step 1: 使用 Demucs 提取人聲，並透過 OpenAI Whisper 將音訊轉錄為歌詞 (SRT/TXT)。

Step 2: 透過 Groq API (Llama-3.1) 將歌詞時間軸轉換為連續且具故事性的影像提示詞。

Step 3: 使用 Stable Diffusion XL (SDXL) 根據提示詞生成高品質影像。

Step 4: 利用 Stable Video Diffusion (SVD) 將靜態影像轉化為短影片片段。

Step 5: 透過 MoviePy 將所有片段與原始音訊進行串接，生成最終的完整 MV。

📂 資料夾結構
為了確保系統正常運行，請維持以下結構。模型檔案需放置於 local_models 目錄下以供系統自動載入。

Plaintext

/project_root
│
├── app.py                      # Flask 後端主程式
├── index.html                   # 前端控制面板 (Web UI)
│
├── audio_processor.py           # Step 1: 音訊與歌詞處理模組
├── prompt_generation_module.py  # Step 2: Groq Prompt 生成模組
├── image_generation_module.py   # Step 3: SDXL 圖像生成模組
├── video_generation_module.py   # Step 4: SVD 影片生成模組
├── video_merger_module.py       # Step 5: 影片合併模組 (無字幕優化版)
│
├── local_models/                # 🤖 核心模型存放處 (手動建立或自動下載)
│   ├── sdxl_base/               # SDXL 模型權重
│   └── svd_base/                # SVD 影片生成模型權重
│
├── uploads/                     # 暫存上傳檔案路徑 (系統自動建立)
├── outputs/                     # 存放生成的影像、影片與 ZIP 包 (系統自動建立)
├── mock_image.png               # 系統初始化測試用圖檔
└──test_resources.zip            # 解壓縮後可以供各step的測試操作，連結:https://drive.google.com/file/d/17S1L5qSi2Fkk-q8iavGAUXIiOv6CYWbC/view?usp=drive_link

🛠️ 安裝環境與套件
本專案建議使用 Conda 管理環境。

1. 建立並啟動環境

conda create -n mv_generator python=3.10.19
conda activate mv_generator
2. 安裝系統級工具
本專案依賴 FFmpeg 進行影像編碼，且 MoviePy 需要 ImageMagick 支援。

conda install -c conda-forge ffmpeg imagemagick librosa fonts-conda-ecosystem pillow -y

3. 安裝 Python 依賴套件

### 安裝 Web 框架
pip install flask flask-cors werkzeug

### 安裝 AI 深度學習核心
### CUDA 12.1 (推薦):

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c 

### 安裝模型運行、人聲分離及影片合成所需的 Python 套件。


### 影片處理 (固定 MoviePy 版本以確保穩定性)
pip install moviepy==1.0.3 imageio-ffmpeg imageio torchcodec soundfile

### AI 生成框架
pip install diffusers==0.30.0 transformers accelerate huggingface-hub
## 模型連結(需下載，可從此直接下載，或是從hugging face官方下載)
## 第一次啟動svd時需要登錄hugging face帳號，取得認證
## https://drive.google.com/drive/folders/1e1eksqjRddjx9GSQ07dudpjKfnmwD532?usp=drive_link

### 音訊轉錄與人聲分離
pip install -U openai-whisper demucs groq
### 生成提示詞
pip install groq
##### 要設定好GROQ_API_KEY = os.getenv("GROQ_API_KEY") (在prompt_generation_module 第28行)

🚀 快速啟動
啟動後端:

python app.py
啟動後，系統會自動檢查並預載入模型 (初次執行會從 Hugging Face 下載)。

啟動前端: 
python -m http.server 8000
使用任意 Web Server 前往 http://localhost:8000/index.html 即可開始操作

操作流程: 按照介面上的 Step 1 至 Step 5 循序漸進操作即可。

⚠️ 注意事項
硬體需求: 建議使用具有 12GB 以上 VRAM 的 NVIDIA 顯卡 (例如 RTX 3060/4070 以上) 以執行 SVD 模型。

API Key: 使用 Step 2 前，請確保 prompt_generation_module.py 中的 GROQ_API_KEY 已正確設定。

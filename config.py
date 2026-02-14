"""全局常量与配置"""

import os

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff", ".aif"
}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"
}

LANG_NAMES: dict[str, str] = {
    "zh": "简体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "auto": "自动检测",
}

# ISO 639-1 → ISO 639-2/B
FFMPEG_LANG_CODES: dict[str, str] = {
    "zh": "chi",
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
}

VAD_THRESHOLD = 0.25          # 语音概率阈值，越低越敏感
VAD_THRESHOLD_DENOISE = 0.35  # 去噪后使用较高阈值
VAD_MIN_SILENCE_MS = 500      # 最短静音时长(ms)
VAD_MIN_SPEECH_MS = 50        # 最短语音时长(ms)
VAD_SPEECH_PAD_MS = 300       # 语音片段前后填充(ms)

DENOISE_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
DENOISE_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "audio-separator-models")

TRANSLATE_BATCH_SIZE = 50     # 每批翻译的字幕条数
TRANSLATE_MAX_RETRIES = 5     # 最大重试次数
TRANSLATE_RETRY_DELAY = 1     # 重试间隔(秒)
TRANSLATE_API_TIMEOUT = 200   # 请求超时(秒)

TranslateConfig = tuple[str, str, str]  # (api_key, base_url, model_name)

SAMPLE_RATE = 16000           # VAD 和 Whisper 采样率
PROGRESS_INTERVAL = 5         # 每 N 个片段打印一次进度

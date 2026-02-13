# MlxVadSRT - High-Performance Speech-to-Text Tool

[中文文档](README.md)

> [!CAUTION]
> **Warning**: This project only supports **macOS (Apple Silicon)** environments (such as M1, M2, M3 chips). `mlx-whisper` relies on Apple Silicon hardware acceleration and cannot run on Intel Macs, Windows, or Linux.

`MlxVadSRT` is a high-performance speech-to-text tool based on MLX Whisper and Silero VAD, optimized for macOS (Apple Silicon). It supports automatic video audio extraction, Voice Activity Detection (VAD), and generation of standard SRT subtitle files.

## Key Features

-   **MLX Hardware Acceleration**: Leveraging Apple Silicon's unified memory architecture and Metal acceleration, `mlx-whisper` not only outperforms standard CPU inference by a wide margin but also rivals the performance of `whisper.cpp`, combining high efficiency with Python's ease of use.
-   **VAD Smart Filtering**: By using `Silero VAD` to pre-detect and extract only speech segments, it not only boosts efficiency by skipping silence but also effectively prevents Whisper from hallucinating during silent periods, significantly improving transcription accuracy.
-   **Subtitle Translation**: Supports translating transcribed subtitles to a specified language, using OpenAI-compatible APIs with automatic fallback to local Ollama.
-   **Standalone Translation Mode**: Supports directly translating existing SRT subtitle files without re-transcription.
-   **Soft Subtitle Embedding**: Supports embedding SRT subtitles as soft subtitles (not hardcoded) into video files using ffmpeg muxing, without re-encoding the video.

## 1. Environment Setup

Before getting started, please ensure the following components are installed on your macOS system:

1.  **Python 3.9+**: Version `3.10` or above is recommended.
2.  **FFmpeg**: Used for audio decoding and extraction.
    ```bash
    brew install ffmpeg
    ```
3.  **Apple Silicon (M1/M2/M3)**: `mlx-whisper` requires Apple Silicon support.
4.  **Memory**:
    *   **16GB+ Unified Memory**: Recommended `mlx-community/whisper-large-v3-mlx`.
    *   **8GB Unified Memory**: Recommended `mlx-community/whisper-large-v3-turbo`.

## 2. Basic Environment Installation

It is recommended to install dependencies in a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate

# Install core dependencies
pip install torch numpy mlx-whisper
```

---

## 3. Deployment Methods

### Method 1: Alias Deployment - (Recommended)

Set up an alias in your terminal configuration file to directly call the Python script. Suitable for scenarios where you frequently modify the script or use it on a fixed machine.

1.  **Modify terminal configuration** (using `zsh` as example):
    ```bash
    nano ~/.zshrc
    ```

2.  **Add alias** (replace with your actual path):
    ```bash
    # Assuming the project is in ~/opt directory
    alias mlxvad='~/opt/venv/bin/python3 ~/opt/transcribe.py'
    ```

3.  **Apply configuration**:
    ```bash
    source ~/.zshrc
    ```

### Method 2: PyInstaller

Package the script and its dependencies into a standalone binary file.

> [!NOTE]
> **Note**: Due to the need to include large deep learning libraries like `torch` and `mlx`, the generated binary file will be quite large (typically over 1GB). The packaging process will also consume considerable memory and time.

1.  **Install packaging tool**:
    ```bash
    pip install pyinstaller
    ```

2.  **Execute packaging**:
    ```bash
    pyinstaller --onefile \
                --name mlxvad \
                --collect-all mlx_whisper \
                --collect-all torch \
                transcribe.py
    ```

3.  **Move to system path** (optional):
    ```bash
    sudo cp dist/mlxvad /usr/local/bin/
    ```

---

## 4. Environment Variable Configuration

This project uses environment variables to configure model download sources and translation APIs.

### 4.1. Model Download Acceleration (Hugging Face)

By default, the program automatically uses a mirror site (`hf-mirror.com`) to accelerate model downloads.
If you need to use the official source or a custom mirror, you can set `HF_ENDPOINT`:

```bash
export HF_ENDPOINT="https://huggingface.co"
```

### 4.2. Translation Function (LLM API)

When using the `--to` parameter, the program calls an OpenAI-compatible API. Please set the following environment variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
```

If the above variables are not configured, the program will attempt to fall back to local Ollama (`http://localhost:11434/v1`, model `qwen3:8b`).

---

## 5. Common Command Examples

| Scenario | Command |
| :--- | :--- |
| **Transcribe video (Chinese)** | `mlxvad --video demo.mp4 --lang zh` |
| **Transcribe audio (auto-detect language)** | `mlxvad --audio record.mp3 --lang auto` |
| **Specify output filename** | `mlxvad --audio test.wav --output result.srt` |
| **Use smaller model (faster)** | `mlxvad --audio test.wav --model mlx-community/whisper-tiny-mlx` |
| **Transcribe and translate to Chinese** | `mlxvad --audio lecture.mp3 --lang en --to zh` |
| **Transcribe and translate to English** | `mlxvad --video demo.mp4 --lang ja --to en` |
| **Translate existing subtitle file** | `mlxvad --srt subtitle.srt --to en` |
| **Translate subtitle with custom output** | `mlxvad --srt subtitle.srt --to zh --output translated.srt` |
| **Embed soft subtitles into video** | `mlxvad --embed --video demo.mp4 --srt demo.zh.srt` |

## 6. Parameter Description

- `--audio`: Input audio file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified.
- `--video`: Input video file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified. In embed mode (`--embed`), must be used with `--srt`.
- `--srt`: Input existing SRT subtitle file path. In translation mode, **must be used with `--to`**. In embed mode (`--embed`), must be used with `--video`.
- `--embed`: Embed the subtitle specified by `--srt` as a soft subtitle (not hardcoded) into the video specified by `--video`. Requires both `--video` and `--srt`. Automatically infers language tag from filename (e.g., `demo.zh.srt` → Chinese).
- `--lang`: Specify language (default: `auto` for auto-detection, options: `zh, en, ja, ko, auto`). Only effective in transcription mode.
- `--to`: Translate subtitles to the specified language (default: no translation, options: `zh, en, ja, ko`). Cannot be the same as `--lang` in transcription mode.
- `--model`: MLX model path or HF repository (default: `mlx-community/whisper-large-v3-mlx`). **Note**: Only supports `mlx-community/whisper` series models. Only effective in transcription mode.
- `--output`: Output SRT filename (default: `output.srt`). When used with `--to`, if explicitly specified the translated file is saved to that path; otherwise it is automatically named `originalname.targetlang.srt`.

---

## 7. Usage Examples

### Scenario: Transcribe a Youtube video (downloaded locally)
Suppose you have downloaded a tutorial video named `lecture.mp4` and want to generate Chinese subtitles:

```bash
mlxvad --video lecture.mp4 --lang zh --output lecture.srt
```

### Scenario: Fast transcription (using a smaller model)
If you don't need high accuracy but prioritize speed, you can use the `tiny` model:

```bash
mlxvad --audio meeting_record.m4a --model mlx-community/whisper-tiny-mlx --output fast_result.srt
```

### Scenario: Auto-detect language
If you're unsure about the language in the audio:

```bash
mlxvad --audio interview.wav --lang auto
```

### Scenario: Transcribe a Japanese video and translate to Chinese
```bash
mlxvad --video anime.mp4 --lang ja --to zh --output anime.srt
```
This will generate two files: `anime.original.srt` (original Japanese subtitles) and `anime.srt` (translated Chinese subtitles).

### Scenario: Translate an existing SRT subtitle file
If you already have an English subtitle file and want to translate it to Chinese:
```bash
mlxvad --srt english.srt --to zh
```
The output file is automatically named `english.zh.srt`. You can also use `--output` to specify the output path:
```bash
mlxvad --srt english.srt --to zh --output chinese_subtitle.srt
```

### Scenario: Embed subtitles into video
After transcription and translation, embed Chinese subtitles into the video (soft subtitles, no re-encoding):
```bash
mlxvad --embed --video anime.mp4 --srt anime.srt
```
Subtitles will be embedded directly into `anime.mp4`, and the player can toggle subtitles on/off. If the subtitle filename contains a language suffix (e.g., `anime.zh.srt`), the language tag will be set automatically.

---

## FAQ
- **First run**: The program will automatically download models from Hugging Face, please ensure network connectivity.
- **Offline usage**: Uncomment `os.environ["HF_HUB_OFFLINE"] = "1"` in the script to force using local cache.
- **Translation timeout**: If the translation API responds slowly, you can modify the `TRANSLATE_API_TIMEOUT` constant in the script (default: 200 seconds).
- **Missing dialogue**: If some speech segments are not recognized, try lowering the `VAD_THRESHOLD` value in the script (default: 0.25, minimum: 0.1).

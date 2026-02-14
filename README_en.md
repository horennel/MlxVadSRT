# MlxVadSRT - High-Performance Speech-to-Text Tool

[中文文档](README.md)

> [!CAUTION]
> **Warning**: This project only supports **macOS (Apple Silicon)** environments (such as M1, M2, M3 chips). `mlx-whisper` relies on Apple Silicon hardware acceleration and cannot run on Intel Macs, Windows, or Linux.

`MlxVadSRT` is a high-performance speech-to-text tool based on MLX Whisper and Silero VAD, optimized for macOS (Apple Silicon). It supports automatic video audio extraction, Voice Activity Detection (VAD), and generation of standard SRT subtitle files.

## Key Features

-   **MLX Hardware Acceleration**: Leveraging Apple Silicon's unified memory architecture and Metal acceleration, `mlx-whisper` not only outperforms standard CPU inference by a wide margin but also rivals the performance of `whisper.cpp`, combining high efficiency with Python's ease of use.
-   **VAD Smart Filtering**: By using `Silero VAD` to pre-detect and extract only speech segments, it not only boosts efficiency by skipping silence but also effectively prevents Whisper from hallucinating during silent periods, significantly improving transcription accuracy.
-   **Vocal Extraction (Denoising)**: Optionally uses MDX-NET model to extract vocals, removing background music and sound effects for improved transcription quality on noisy audio.
-   **Subtitle Translation**: Supports translating transcribed subtitles to a specified language, using OpenAI-compatible APIs with automatic fallback to local Ollama.
-   **Standalone Translation Mode**: Supports directly translating existing SRT subtitle files without re-transcription.
-   **Soft Subtitle Embedding**: Supports embedding SRT subtitles as soft subtitles (not hardcoded) into video files using ffmpeg muxing, without re-encoding the video. Outputs to a new file, preserving the original.

## Project Structure

```
MlxVadSRT/
├── main.py          # Entry point: CLI argument parsing and workflow routing
├── config.py        # Global constants and configuration (zero dependencies)
├── utils.py         # Utilities: audio loading, SRT formatting/parsing, file type detection
├── transcribe.py    # Core transcription: VAD segmentation + Whisper segment-by-segment
├── denoise.py       # Vocal extraction: MDX-NET model for removing BGM/SFX
├── translate.py     # Translation module: LLM API batch subtitle translation
└── embed.py         # Subtitle embedding: FFmpeg soft subtitle muxing
```

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

# Optional: install vocal extraction dependency (for --denoise feature)
pip install "audio-separator[cpu]"
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
    # Assuming the project is in ~/opt/MlxVadSRT directory
    alias mlxvad='~/opt/venv/bin/python3 ~/opt/MlxVadSRT/main.py'
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
                main.py
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
| **Basic Transcription (Chinese)** | `mlxvad --video demo.mp4 --lang zh` |
| **Transcribe Audio (Auto-detect)** | `mlxvad --audio record.mp3 --lang auto` |
| **Transcribe + Embed (Soft Subs)** | `mlxvad --video demo.mp4 --embed` |
| **Transcribe + Translate + Embed (One-Stop)** | `mlxvad --video demo.mp4 --to zh --embed` |
| **Pure Translation (Existing SRT)** | `mlxvad --srt source.srt --to zh` |
| **Pure Embed (Existing SRT)** | `mlxvad --embed --video demo.mp4 --srt source.srt` |
| **Translate + Embed (Existing SRT)** | `mlxvad --embed --video demo.mp4 --srt source.srt --to zh` |
| **Denoise then Transcribe (w/ BGM)** | `mlxvad --video movie.mp4 --denoise` |

> **Note**: When using `--denoise`, temporary vocal files are stored in the system's temporary directory (prefixed with `mlxvadsrt_vocals_`). They are automatically cleaned up after the task. If interrupted, you can manually clear this directory.

## 6. Parameter Description

- `--audio`: Input audio file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified.
- `--video`: Input video file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified. In embed mode (`--embed`), it is **required**.
- `--srt`: Input existing SRT subtitle file path. In translation mode, used with `--to`. In embed mode (`--embed`), it is optional (if not provided, auto-generated from video).
- `--embed`: Enable subtitle embedding mode. Embeds generated or specified subtitles as soft subtitles into the video.
    - **With `--video` (no `--srt`)**: Automatically performs [Transcribe -> (Translate) -> Embed] pipeline.
    - **With `--video` + `--srt`**: Embeds the specified subtitle (translates first if `--to` is present).
    - **Output**: Generates a new file (e.g., `demo_embedded.mp4`), preserving the original video. Automatically deletes the intermediate SRT file after success.
- `--lang`: Specify language (default: `auto` for auto-detection, options: `zh, en, ja, ko, auto`). Only effective in transcription mode.
- `--to`: Translate subtitles to the specified language (default: no translation, options: `zh, en, ja, ko`). Cannot be the same as `--lang` in transcription mode.
- `--model`: MLX model path or HF repository (default: `mlx-community/whisper-large-v3-mlx`). **Note**: Only supports `mlx-community/whisper` series models. Only effective in transcription mode.
- `--output`: Output SRT filename (default: follows input filename, e.g., `demo.mp4` → `demo.srt`). When used with `--to`, if explicitly specified the translated file is saved to that path; otherwise it is automatically named `originalname.targetlang.srt`.
- `--denoise`: Extract vocals using MDX-NET model before transcription, removing background music and sound effects. Requires `audio-separator` library. Ideal for movies, TV shows, and variety shows with BGM.

---

## 7. Advanced Usage Examples

### 🚀 Ultimate Usage: Transcribe + Translate + Embed (One-Stop)
Convert raw video directly to a finished video with subtitles:
```bash
mlxvad --video movie.mp4 --lang en --to zh --embed
```
> **Workflow**: Auto-transcribe English -> Auto-translate to Chinese -> Auto-embed Chinese subtitles -> Generate `movie_embedded.mp4` -> Cleanup temporary SRT.

### Scenario: Transcribe + Embed Only (No Translation)
Useful for generating same-language subtitles:
```bash
mlxvad --video lecture.mp4 --embed
```

### Scenario: Translate Existing Subtitle + Embed
If you already have English subtitles `eng.srt` and want to translate to Chinese and embed:
```bash
mlxvad --embed --video movie.mp4 --srt eng.srt --to zh
```

### Scenario: Manual Step-by-Step (Traditional)
If you want to keep intermediate files or manually proofread subtitles before embedding:

1. **Transcribe and Translate**:
   ```bash
   mlxvad --video demo.mp4 --to zh
   # Generates demo.original.srt and demo.zh.srt (keep for proofreading)
   ```

2. **(Optional) Manually edit demo.zh.srt**...

3. **Embed Proofread Subtitles**:
   ```bash
   mlxvad --embed --video demo.mp4 --srt demo.zh.srt
   ```

### Scenario: Denoise + One-Stop Pipeline
For noisy videos with background music:
```bash
mlxvad --video vlog.mp4 --to zh --embed --denoise
```

---

## 8. `--denoise` Recommended Use Cases

| Scenario | Recommend `--denoise`? |
|----------|:---:|
| Movies / TV shows (with soundtrack) | ✅ Recommended |
| Action films (explosions/fights + dialogue) | ✅ Highly recommended |
| Variety shows / reality TV | ✅ Recommended |
| Pure dialogue podcasts / meeting recordings | ❌ Not needed |
| Documentaries (mostly narration) | ⚠️ Depends |

---

## FAQ
- **First run**: The program will automatically download models from Hugging Face, please ensure network connectivity.
- **Offline usage**: Uncomment `os.environ["HF_HUB_OFFLINE"] = "1"` in `main.py` to force using local cache.
- **Translation timeout**: If the translation API responds slowly, modify the `TRANSLATE_API_TIMEOUT` constant in `config.py` (default: 200 seconds).
- **Missing dialogue**: If some speech segments are not recognized, try lowering the `VAD_THRESHOLD` value in `config.py` (default: 0.25, minimum: 0.1).
- **`--denoise` ImportError**: Make sure `audio-separator` is installed: `pip install "audio-separator[cpu]"`.
- **`--denoise` model download fails**: Models are cached at `~/.cache/audio-separator-models/`. You can manually download and place them there.

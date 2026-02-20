# MlxVadSRT - High-Performance Speech-to-Text Tool

[中文文档](README.md)

> [!CAUTION]
> **Warning**: This project only supports **macOS (Apple Silicon)** environments (such as M1, M2, M3, M4 chips).
`mlx-whisper` relies on Apple Silicon hardware acceleration and cannot run on Intel Macs, Windows, or Linux.

`MlxVadSRT` is a high-performance speech-to-text tool based on MLX Whisper and Silero VAD, optimized for macOS (Apple
Silicon). It supports automatic video audio extraction, Voice Activity Detection (VAD), and generation of standard SRT
subtitle files.

## Key Features

- **MLX Hardware Acceleration**: Leveraging Apple Silicon's unified memory architecture and Metal acceleration,
  `mlx-whisper` not only outperforms standard CPU inference by a wide margin but also rivals the performance of
  `whisper.cpp`, combining high efficiency with Python's ease of use.
- **VAD Smart Filtering**: By using `Silero VAD` to pre-detect and extract only speech segments, it not only boosts
  efficiency by skipping silence but also effectively prevents Whisper from hallucinating during silent periods,
  significantly improving transcription accuracy.
- **Vocal Extraction (Denoising)**: Optionally uses MDX-NET model to extract vocals, removing background music and sound
  effects for improved transcription quality on noisy audio.
- **Subtitle Translation**: Supports translating transcribed subtitles to a specified language, using OpenAI-compatible
  APIs with multi-threaded concurrent translation and automatic fallback to local Ollama.
- **Standalone Translation Mode**: Supports directly translating existing SRT subtitle files without re-transcription.
- **Soft Subtitle Embedding**: Supports embedding SRT subtitles as soft subtitles (not hardcoded) into video files using
  ffmpeg muxing, without re-encoding the video. Outputs to a new file, preserving the original.
- **Web Graphical Interface**: Provides a minimalist web console based on Gradio, supporting task cancellation,
  real-time stream logging, and processing progress display.

## 1. Environment Setup

Before getting started, please ensure the following components are installed on your macOS system:

1. **Python 3.9+**: Version `3.10` or above is recommended.
2. **FFmpeg**: Used for audio decoding and extraction.
   ```bash
   brew install ffmpeg
   ```
3. **Apple Silicon (M1/M2/M3/M4)**: `mlx-whisper` requires Apple Silicon support.
4. **Memory**:
    * **16GB+ Unified Memory**: Recommended `mlx-community/whisper-large-v3-mlx`.
    * **8GB Unified Memory**: Recommended `mlx-community/whisper-large-v3-turbo`.

## 2. Basic Environment Installation

It is recommended to install dependencies in a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

`requirements.txt` includes the following dependencies:

- `mlx-whisper` — MLX Whisper transcription engine
- `numpy` / `torch` — Numerical computation and VAD model
- `gradio` — Web UI graphical interface
- `audio-separator[cpu]` — Vocal extraction (optional)

---

## 3. Deployment Methods

### Method 1: Web Interface Mode (Recommended)

Run the following command directly to open the full operation interface in your browser:

```bash
python main.py --web
```

Or, if you configured an alias below, you can launch it with a single click using `mlxvad --web`.

This interface supports:

- Uploading audio/video/SRT files
- Selecting source language and target translation language
- Checking denoising and subtitle embedding features
- Real-time viewing of operation logs and extracting translation results

### Method 2: Command Line Mode

#### 1. Configure System Alias (Recommended)

Set up an alias in your terminal configuration file to directly call the Python script. Suitable for use across any
directories.

1. **Modify terminal configuration** (using `zsh` and `conda` as an example):
   ```bash
   nano ~/.zshrc
   ```

2. **Add alias** (replace with your actual path and environment name):
   ```bash
   # Assuming the project is in ~/opt/MlxVadSRT with conda environment mlx
   alias mlxvad='conda run -n mlx python ~/opt/MlxVadSRT/main.py'
   ```

3. **Apply configuration**:
   ```bash
   source ~/.zshrc
   ```

**From then on, you can input `mlxvad --web` in any terminal to launch the web interface, or follow the example command
line instructions below:**

#### 2. Basic Command Line Usage

Run tasks directly from the command line. To see all supported arguments:

```bash
python main.py --help
```

## 4. Environment Variable Configuration

This project uses environment variables to configure model download sources and translation APIs.

### 4.1. Model Download Acceleration (Hugging Face)

By default, the program automatically uses a mirror site (`hf-mirror.com`) to accelerate model downloads.
If you need to use the official source or a custom mirror, you can set `HF_ENDPOINT`:

```bash
export HF_ENDPOINT="https://huggingface.co"
```

### 4.2. Translation Function (LLM API)

When using the `--to` parameter, the program calls an OpenAI-compatible API. Please set the following environment
variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
```

If the above variables are not configured, the program will attempt to fall back to local Ollama (
`http://localhost:11434/v1`, model `qwen3:8b`).

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

> **Note**: When using `--denoise`, temporary vocal files are stored in the system's temporary directory (prefixed with
`mlxvadsrt_vocals_`). They are automatically cleaned up after the task. If interrupted, you can manually clear this
directory.

## 6. Parameter Description

- `--audio`: Input audio file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified.
- `--video`: Input video file path. In transcription mode, only one of `--audio`, `--video`, `--srt` can be specified.
  In embed mode (`--embed`), it is **required**.
- `--srt`: Input existing SRT subtitle file path. In translation mode, used with `--to`. In embed mode (`--embed`), it
  is optional (if not provided, auto-generated from video).
- `--embed`: Enable subtitle embedding mode. Embeds generated or specified subtitles as soft subtitles into the video.
    - **With `--video` (no `--srt`)**: Automatically performs [Transcribe -> (Translate) -> Embed] pipeline.
    - **With `--video` + `--srt`**: Embeds the specified subtitle (translates first if `--to` is present).
    - **Output**: Generates a new file (e.g., `demo_embedded.mp4`), preserving the original video. Automatically deletes
      the intermediate SRT file after success.
- `--lang`: Specify language (default: `auto` for auto-detection, options: `zh, en, ja, ko, auto`). Only effective in
  transcription mode.
- `--to`: Translate subtitles to the specified language (default: no translation, options: `zh, en, ja, ko`). Cannot be
  the same as `--lang` in transcription mode.
- `--model`: MLX model path or HF repository (default: `mlx-community/whisper-large-v3-mlx`). **Note**: Only supports
  `mlx-community/whisper` series models. Only effective in transcription mode.
- `--output`: Output SRT filename (default: follows input filename, e.g., `demo.mp4` → `demo.srt`). When used with
  `--to`, if explicitly specified the translated file is saved to that path; otherwise it is automatically named
  `originalname.targetlang.srt`.
- `--denoise`: Extract vocals using MDX-NET model before transcription, removing background music and sound effects.
  Requires `audio-separator` library. Ideal for movies, TV shows, and variety shows with BGM.

## FAQ

- **First run**: The program will automatically download models from Hugging Face, please ensure network connectivity.
- **Offline usage**: Uncomment `os.environ["HF_HUB_OFFLINE"] = "1"` in `main.py` to force using local cache.
- **Translation timeout**: If the translation API responds slowly, modify the `TRANSLATE_API_TIMEOUT` constant in
  `core/config.py` (default: 200 seconds).
- **Missing dialogue**: If some speech segments are not recognized, try lowering the `VAD_THRESHOLD` value in
  `core/config.py` (default: 0.25, minimum: 0.1).
- **Over-filtering after denoising**: If using `--denoise` causes dialogue to be lost, try lowering the
  `VAD_THRESHOLD_DENOISE` value in `core/config.py` (default: 0.35).
- **`--denoise` ImportError**: Make sure `audio-separator` is installed: `pip install "audio-separator[cpu]"`.
- **`--denoise` model download fails**: Models are cached at `~/.cache/audio-separator-models/`. You can manually
  download and place them there.

# MlxVadSRT - High-Performance Speech-to-Text Tool

[中文文档](README.md)

> [!CAUTION]
> **Warning**: This project only supports **macOS (Apple Silicon)** environments (such as M1, M2, M3 chips). `mlx-whisper` relies on Apple Silicon hardware acceleration and cannot run on Intel Macs, Windows, or Linux.

`MlxVadSRT` is a high-performance speech-to-text tool based on MLX Whisper and Silero VAD, optimized for macOS (Apple Silicon). It supports automatic video audio extraction, Voice Activity Detection (VAD), and generation of standard SRT subtitle files.

## 1. Environment Setup

Before getting started, please ensure the following components are installed on your macOS system:

1.  **Python 3.9+**: Version `3.10` or above is recommended.
2.  **FFmpeg**: Used for audio decoding and extraction.
    ```bash
    brew install ffmpeg
    ```
3.  **Apple Silicon (M1/M2/M3)**: `mlx-whisper` requires Apple Silicon support.

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

## 4. Common Command Examples

| Scenario | Command |
| :--- | :--- |
| **Transcribe video (Chinese)** | `mlxvad --video demo.mp4 --lang zh` |
| **Transcribe audio (auto-detect language)** | `mlxvad --audio record.mp3 --lang auto` |
| **Specify output filename** | `mlxvad --audio test.wav --output result.srt` |
| **Use smaller model (faster)** | `mlxvad --audio test.wav --model mlx-community/whisper-tiny-mlx` |

## 5. Parameter Description

- `--audio`: Input audio file path. **Mutually exclusive** with `--video`, only one can be specified.
- `--video`: Input video file path. **Mutually exclusive** with `--audio`, only one can be specified.
- `--lang`: Specify language (default: `auto` for auto-detection, options: `zh, en, ja, ko, auto`).
- `--model`: MLX model path or HF repository (default: `mlx-community/whisper-large-v3-mlx`). **Note**: Only supports `mlx-community/whisper` series models.
- `--output`: Output SRT filename (default: `output.srt`).
- `--sample_rate`: Sample rate (default: `16000`).

---

## 6. Usage Examples

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

---

## FAQ
- **First run**: The program will automatically download models from Hugging Face, please ensure network connectivity.
- **Offline usage**: Uncomment `os.environ["HF_HUB_OFFLINE"] = "1"` in the script to force using local cache.

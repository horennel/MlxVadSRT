# MlxVadSRT - 高性能语音转文字工具

> [!CAUTION]
> **警告**：本项目仅支持 **macOS (Apple Silicon)** 环境（如 M1, M2, M3 芯片）。`mlx-whisper` 依赖 Apple 芯片的硬件加速，无法在 Intel Mac、Windows 或 Linux 上运行。

`MlxVadSRT` (原 `transcribe.py`) 是一个基于 MLX Whisper 和 Silero VAD 的高性能语音转文字工具，专为 macOS (Apple Silicon) 优化。它支持自动提取视频音频、人声检测（VAD）以及生成标准 SRT 字幕文件。

## 1. 环境准备

在开始之前，请确保您的 macOS 系统已安装以下基础组件：

1.  **Python 3.9+**: 建议使用 `3.10` 或以上版本。
2.  **FFmpeg**: 用于音频解码和提取。
    ```bash
    brew install ffmpeg
    ```
3.  **Apple Silicon (M1/M2/M3)**: `mlx-whisper` 需要 Apple 芯片支持。

## 2. 基础环境安装

建议在虚拟环境中安装依赖：

```bash
# 创建虚拟环境
python3 -m venv venv
# 激活虚拟环境
source venv/bin/activate

# 安装核心依赖
pip install torch numpy mlx-whisper
```

---

## 3. 部署方式

### 方式一：Alias (别名部署) - （推荐）

通过在终端配置文件中设置别名，直接调用 Python 脚本。适合经常需要修改脚本或在固定机器上使用的场景。

1.  **修改终端配置**（以 `zsh` 为例）：
    ```bash
    nano ~/.zshrc
    ```

2.  **添加别名**（请根据实际路径替换）：
    ```bash
    # 假设项目在 ~/opt 目录下
    alias transcribe='~/opt/venv/bin/python3 ~/opt/transcribe.py'
    # 或者如果您想使用新名字作为命令：
    alias mlxvad='~/opt/venv/bin/python3 ~/opt/transcribe.py'
    ```

3.  **生效配置**：
    ```bash
    source ~/.zshrc
    ```

### 方式二：PyInstaller

将脚本及其依赖打包成一个独立的二进制文件。

1.  **安装打包工具**：
    ```bash
    pip install pyinstaller
    ```

2.  **执行打包**：
    ```bash
    pyinstaller --onefile \
                --name mlxvad \
                --collect-all mlx_whisper \
                --collect-all torch \
                transcribe.py
    ```

3.  **移动到系统路径**（可选）：
    ```bash
    sudo cp dist/mlxvad /usr/local/bin/
    ```

---

## 4. 常用命令示例

| 场景 | 命令 |
| :--- | :--- |
| **转录视频 (中文)** | `transcribe --video demo.mp4 --lang zh` |
| **转录音频 (自动检测语言)** | `transcribe --audio record.mp3 --lang auto` |
| **指定输出文件名** | `transcribe --audio test.wav --output result.srt` |
| **使用更小的模型 (加速)** | `transcribe --audio test.wav --model mlx-community/whisper-tiny-mlx` |

## 5. 参数说明

- `--audio`: 输入音频文件路径。
- `--video`: 输入视频文件路径。
- `--lang`: 指定语言 (默认: `zh`, 可选: `zh, en, ja, ko, auto`)。
- `--model`: MLX 模型路径或 HF 仓库 (默认: `mlx-community/whisper-large-v3-mlx`)。
- `--output`: 输出 SRT 文件名 (默认: `output.srt`)。
- `--sample_rate`: 采样率 (默认: `16000`)。

---

## 常见问题
- **首次运行**: 程序会自动从 Hugging Face 下载模型，请保持网络畅通。
- **离线使用**: 设置 `os.environ["HF_HUB_OFFLINE"] = "1"`（已在脚本中默认开启）以强制使用本地缓存。

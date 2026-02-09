# MlxVadSRT - 高性能语音转文字工具

[English Documentation](README_en.md)

> [!CAUTION]
> **警告**：本项目仅支持 **macOS (Apple Silicon)** 环境（如 M1, M2, M3 芯片）。`mlx-whisper` 依赖 Apple 芯片的硬件加速，无法在 Intel Mac、Windows 或 Linux 上运行。

`MlxVadSRT` 是一个基于 MLX Whisper 和 Silero VAD 的高性能语音转文字工具，专为 macOS (Apple Silicon) 优化。它支持自动提取视频音频、人声检测（VAD）以及生成标准 SRT 字幕文件。

## 核心特色

-   **MLX 硬件加速**：利用 Apple Silicon 的统一内存架构和 Metal 加速，`mlx-whisper` 不仅速度远超普通 CPU 推理，更能媲美 `whisper.cpp`，兼顾高性能与 Python 生态的易用性。
-   **VAD 智能过滤**：通过 `Silero VAD` 预先检测并仅提取人声片段，不仅能跳过静音区域大幅提升效率，更能有效避免 Whisper 在静音片段产生幻觉（Hallucination），显著提升转录准确度。

## 1. 环境准备

在开始之前，请确保您的 macOS 系统已安装以下基础组件：

1.  **Python 3.9+**: 建议使用 `3.10` 或以上版本。
2.  **FFmpeg**: 用于音频解码和提取。
    ```bash
    brew install ffmpeg
    ```
3.  **Apple Silicon (M1/M2/M3)**: `mlx-whisper` 需要 Apple 芯片支持。
4.  **内存建议**:
    *   **16GB+ 统一内存**: 推荐使用 `mlx-community/whisper-large-v3-mlx`。
    *   **8GB 统一内存**: 推荐使用 `mlx-community/whisper-large-v3-turbo`。

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
    alias mlxvad='~/opt/venv/bin/python3 ~/opt/transcribe.py'
    ```

3.  **生效配置**：
    ```bash
    source ~/.zshrc
    ```

### 方式二：PyInstaller

将脚本及其依赖打包成一个独立的二进制文件。

> [!NOTE]
> **注意**：由于需要包含 `torch` 和 `mlx` 等大型深度学习库，生成的二进制文件体积会比较大（通常在 1GB 以上）。打包过程也会占用较多内存和时间。

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
| **转录视频 (中文)** | `mlxvad --video demo.mp4 --lang zh` |
| **转录音频 (自动检测语言)** | `mlxvad --audio record.mp3 --lang auto` |
| **指定输出文件名** | `mlxvad --audio test.wav --output result.srt` |
| **使用更小的模型 (加速)** | `mlxvad --audio test.wav --model mlx-community/whisper-tiny-mlx` |

## 5. 参数说明

- `--audio`: 输入音频文件路径。与 `--video` **互斥**，只能指定其中一个。
- `--video`: 输入视频文件路径。与 `--audio` **互斥**，只能指定其中一个。
- `--lang`: 指定语言 (默认: `auto` 自动检测, 可选: `zh, en, ja, ko, auto`)。
- `--model`: MLX 模型路径或 HF 仓库 (默认: `mlx-community/whisper-large-v3-mlx`)。**注意**：仅支持 `mlx-community/whisper` 系列模型。
- `--output`: 输出 SRT 文件名 (默认: `output.srt`)。
- `--sample_rate`: 采样率 (默认: `16000`)。

---

## 6. 使用示例

### 场景：转录一个 Youtube 视频（已下载到本地）
假设你下载了一个名为 `lecture.mp4` 的教程视频，想生成中文字幕：

```bash
mlxvad --video lecture.mp4 --lang zh --output lecture.srt
```

### 场景：快速转录（使用更小的模型）
如果你对精度要求不高，但追求速度，可以使用 `tiny` 模型：

```bash
mlxvad --audio meeting_record.m4a --model mlx-community/whisper-tiny-mlx --output fast_result.srt
```

### 场景：自动检测语言
如果你不确定音频中的语言：

```bash
mlxvad --audio interview.wav --lang auto
```

---

## 常见问题
- **首次运行**: 程序会自动从 Hugging Face 下载模型，请保持网络畅通。
- **离线使用**: 在脚本中取消 `os.environ["HF_HUB_OFFLINE"] = "1"` 的注释，可强制使用本地缓存。

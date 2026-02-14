# MlxVadSRT - 高性能语音转文字工具

[English Documentation](README_en.md)

> [!CAUTION]
> **警告**：本项目仅支持 **macOS (Apple Silicon)** 环境（如 M1, M2, M3 芯片）。`mlx-whisper` 依赖 Apple 芯片的硬件加速，无法在 Intel Mac、Windows 或 Linux 上运行。

`MlxVadSRT` 是一个基于 MLX Whisper 和 Silero VAD 的高性能语音转文字工具，专为 macOS (Apple Silicon) 优化。它支持自动提取视频音频、人声检测（VAD）以及生成标准 SRT 字幕文件。

## 核心特色

-   **MLX 硬件加速**：利用 Apple Silicon 的统一内存架构和 Metal 加速，`mlx-whisper` 不仅速度远超普通 CPU 推理，更能媲美 `whisper.cpp`，兼顾高性能与 Python 生态的易用性。
-   **VAD 智能过滤**：通过 `Silero VAD` 预先检测并仅提取人声片段，不仅能跳过静音区域大幅提升效率，更能有效避免 Whisper 在静音片段产生幻觉（Hallucination），显著提升转录准确度。
-   **人声提取 (去噪)**：可选使用 MDX-NET 模型提取人声，去除背景音乐和音效，进一步提升嘈杂音频的转录质量。
-   **字幕翻译**：支持将转录后的字幕翻译为指定语言，使用 OpenAI 兼容 API，支持自动回退到本地 Ollama。
-   **独立翻译模式**：支持直接翻译已有的 SRT 字幕文件，无需重新转录。
-   **软字幕嵌入**：支持将 SRT 字幕作为软字幕（非硬编码）嵌入视频文件，使用 ffmpeg 封装，不重新编码视频。输出到新文件，不覆盖原视频。

## 项目结构

```
MlxVadSRT/
├── main.py          # 入口：命令行参数解析与主流程调度
├── config.py        # 全局常量与配置（零依赖）
├── utils.py         # 通用工具：音频读取、SRT 格式化/解析、文件类型检测
├── transcribe.py    # 核心转录：VAD 分段 + Whisper 逐段转录
├── denoise.py       # 人声提取：MDX-NET 模型去除 BGM/音效
├── translate.py     # 翻译模块：LLM API 批量翻译字幕
└── embed.py         # 字幕嵌入：FFmpeg 软字幕封装
```

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

# 可选：安装人声提取依赖 (用于 --denoise 功能)
pip install "audio-separator[cpu]"
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
    # 假设项目在 ~/opt/MlxVadSRT 目录下
    alias mlxvad='~/opt/venv/bin/python3 ~/opt/MlxVadSRT/main.py'
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
                main.py
    ```

3.  **移动到系统路径**（可选）：
    ```bash
    sudo cp dist/mlxvad /usr/local/bin/
    ```

---

## 4. 环境变量配置

本项目使用环境变量来配置模型下载源和翻译 API。

### 4.1. 模型下载加速 (Hugging Face)

默认情况下，程序会自动使用国内镜像站 (`hf-mirror.com`) 加速模型下载。
如果需使用官方源或自定义镜像，可设置 `HF_ENDPOINT`：

```bash
export HF_ENDPOINT="https://huggingface.co"
```

### 4.2. 翻译功能 (LLM API)

使用 `--to` 参数时，程序会调用 OpenAI 兼容 API。请设置以下环境变量：

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
```

如果未配置上述变量，程序将尝试回退到本地 Ollama (`http://localhost:11434/v1`, 模型 `qwen3:8b`)。

---

## 5. 常用命令示例

| 场景 | 命令 |
| :--- | :--- |
| **转录视频 (中文)** | `mlxvad --video demo.mp4 --lang zh` |
| **转录音频 (自动检测语言)** | `mlxvad --audio record.mp3 --lang auto` |
| **指定输出文件名** | `mlxvad --audio test.wav --output result.srt` |
| **使用更小的模型 (加速)** | `mlxvad --audio test.wav --model mlx-community/whisper-tiny-mlx` |
| **转录并翻译为中文** | `mlxvad --audio lecture.mp3 --lang en --to zh` |
| **转录并翻译为英文** | `mlxvad --video demo.mp4 --lang ja --to en` |
| **去噪后转录 (有 BGM)** | `mlxvad --video movie.mp4 --lang zh --denoise` |
| **去噪 + 转录 + 翻译** | `mlxvad --video movie.mkv --lang en --to zh --denoise` |
| **翻译已有字幕文件** | `mlxvad --srt subtitle.srt --to en` |
| **翻译已有字幕并指定输出** | `mlxvad --srt subtitle.srt --to zh --output translated.srt` |
| **嵌入软字幕到视频** | `mlxvad --embed --video demo.mp4 --srt demo.zh.srt` |

## 6. 参数说明

- `--audio`: 输入音频文件路径。转录模式下与 `--video`、`--srt` 只能指定其中一个。
- `--video`: 输入视频文件路径。转录模式下与 `--audio`、`--srt` 只能指定其中一个。嵌入模式（`--embed`）下需配合 `--srt` 使用。
- `--srt`: 输入已有 SRT 字幕文件路径。翻译模式下**必须配合 `--to` 使用**；嵌入模式（`--embed`）下需配合 `--video` 使用。
- `--embed`: 将 `--srt` 指定的字幕作为软字幕（非硬编码）嵌入 `--video` 指定的视频。需同时指定 `--video` 和 `--srt`。输出为新文件（如 `demo_embedded.mp4`），不覆盖原视频。嵌入成功后自动删除源 SRT 文件。自动从文件名推断语言标签（如 `demo.zh.srt` → 中文）。
- `--lang`: 指定语言 (默认: `auto` 自动检测, 可选: `zh, en, ja, ko, auto`)。仅在转录模式下有效。
- `--to`: 将字幕翻译为指定语言 (默认: 不翻译, 可选: `zh, en, ja, ko`)。转录模式下不能与 `--lang` 相同。
- `--model`: MLX 模型路径或 HF 仓库 (默认: `mlx-community/whisper-large-v3-mlx`)。**注意**：仅支持 `mlx-community/whisper` 系列模型。仅在转录模式下有效。
- `--output`: 输出 SRT 文件名 (默认: 跟随输入文件名，如 `demo.mp4` → `demo.srt`)。配合 `--to` 使用时，显式指定则翻译文件保存到该路径，未指定则自动命名为 `原文件名.目标语言.srt`。
- `--denoise`: 转录前先用 MDX-NET 模型提取人声，去除背景音乐和音效。需额外安装 `audio-separator` 库。适用于电影、电视剧、综艺等有 BGM 的场景。

---

## 7. 使用示例

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

### 场景：转录日语视频并翻译为中文
```bash
mlxvad --video anime.mp4 --lang ja --to zh --output anime.srt
```
执行后会生成两个文件：`anime.original.srt`（原始日语字幕）和 `anime.srt`（翻译后的中文字幕）。

### 场景：去噪后转录（有背景音乐的电影/电视剧）
对于有大量 BGM 和音效的视频，使用 `--denoise` 可以显著提升转录质量：

```bash
mlxvad --video movie.mp4 --lang zh --denoise
```

处理流程：先用 MDX-NET 提取人声 → 再用 VAD 检测语音片段 → 最后用 Whisper 逐段转录。模型会自动下载到 `~/.cache/audio-separator-models/`，首次使用约需下载 100-200MB。

### 场景：去噪 + 转录 + 翻译一条龙
```bash
mlxvad --video movie.mkv --lang en --to zh --denoise
```

### 场景：翻译已有的 SRT 字幕文件
如果你已经有一份英文字幕，想翻译为中文：
```bash
mlxvad --srt english.srt --to zh
```
输出文件自动命名为 `english.zh.srt`。也可以用 `--output` 指定输出路径：
```bash
mlxvad --srt english.srt --to zh --output chinese_subtitle.srt
```

### 场景：将字幕嵌入视频
转录并翻译完成后，将中文字幕嵌入视频（软字幕，不重新编码）：
```bash
mlxvad --embed --video anime.mp4 --srt anime.srt
```
字幕会嵌入到新文件 `anime_embedded.mp4` 中（不覆盖原视频），嵌入成功后源 SRT 文件会被自动删除。如果字幕文件名包含语言后缀（如 `anime.zh.srt`），语言标签会自动设置。

---

## 8. `--denoise` 适用场景建议

| 场景 | 是否推荐 `--denoise` |
|------|:---:|
| 电影/电视剧（有配乐） | ✅ 推荐 |
| 动作片（爆炸/打斗 + 对白） | ✅ 强烈推荐 |
| 综艺/真人秀 | ✅ 推荐 |
| 纯对话播客/会议录音 | ❌ 不需要 |
| 纪录片（旁白为主） | ⚠️ 看情况 |

---

## 常见问题
- **首次运行**: 程序会自动从 Hugging Face 下载模型，请保持网络畅通。
- **离线使用**: 在 `main.py` 中取消 `os.environ["HF_HUB_OFFLINE"] = "1"` 的注释，可强制使用本地缓存。
- **翻译超时**: 如果翻译 API 响应较慢，可修改 `config.py` 中的 `TRANSLATE_API_TIMEOUT` 常量（默认 200 秒）。
- **台词遗漏**: 如果发现部分语音未被识别到，可降低 `config.py` 中的 `VAD_THRESHOLD` 值（默认 0.25，最低 0.1）。
- **`--denoise` 报 ImportError**: 请确保安装了 `audio-separator` 库：`pip install "audio-separator[cpu]"`。
- **`--denoise` 模型下载失败**: 模型缓存在 `~/.cache/audio-separator-models/`，可手动下载后放入该目录。

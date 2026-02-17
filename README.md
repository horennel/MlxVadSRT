# MlxVadSRT - 高性能语音转文字工具

[English Documentation](README_en.md)

> [!CAUTION]
> **警告**：本项目仅支持 **macOS (Apple Silicon)** 环境（如 M1, M2, M3, M4 芯片）。`mlx-whisper` 依赖 Apple 芯片的硬件加速，无法在 Intel Mac、Windows 或 Linux 上运行。

`MlxVadSRT` 是一个基于 MLX Whisper 和 Silero VAD 的高性能语音转文字工具，专为 macOS (Apple Silicon) 优化。它支持自动提取视频音频、人声检测（VAD）以及生成标准 SRT 字幕文件。

## 核心特色

-   **MLX 硬件加速**：利用 Apple Silicon 的统一内存架构和 Metal 加速，`mlx-whisper` 不仅速度远超普通 CPU 推理，更能媲美 `whisper.cpp`，兼顾高性能与 Python 生态的易用性。
-   **VAD 智能过滤**：通过 `Silero VAD` 预先检测并仅提取人声片段，不仅能跳过静音区域大幅提升效率，更能有效避免 Whisper 在静音片段产生幻觉（Hallucination），显著提升转录准确度。
-   **人声提取 (去噪)**：可选使用 MDX-NET 模型提取人声，去除背景音乐和音效，进一步提升嘈杂音频的转录质量。
-   **字幕翻译**：支持将转录后的字幕翻译为指定语言，使用 OpenAI 兼容 API，支持多线程并发翻译，并可自动回退到本地 Ollama。
-   **独立翻译模式**：支持直接翻译已有的 SRT 字幕文件，无需重新转录。
-   **软字幕嵌入**：支持将 SRT 字幕作为软字幕（非硬编码）嵌入视频文件，使用 ffmpeg 封装，不重新编码视频。输出到新文件，不覆盖原视频。
-   **GUI 图形界面**：提供基于 customtkinter 的原生 macOS 风格图形界面，支持任务取消、实时日志、窗口位置记忆。

## 项目结构

```
MlxVadSRT/
├── main.py              # CLI 入口：命令行参数解析与主流程调度
├── gui_main.py          # GUI 入口：启动图形界面（含 PATH 同步）
├── setup.py             # py2app 打包配置
├── requirements.txt     # Python 依赖列表
├── assets/
│   └── MlxVadSRT.icns   # macOS App 图标
├── core/                # 核心业务模块
│   ├── __init__.py
│   ├── config.py        # 全局常量与配置（零依赖）
│   ├── utils.py         # 通用工具：音频读取、SRT 格式化/解析、文件类型检测
│   ├── transcribe.py    # 核心转录：VAD 分段 + Whisper 逐段转录
│   ├── denoise.py       # 人声提取：MDX-NET 模型去除 BGM/音效
│   ├── translate.py     # 翻译模块：LLM API 多线程批量翻译字幕
│   └── embed.py         # 字幕嵌入：FFmpeg 软字幕封装
└── gui/                 # GUI 模块
    ├── __init__.py
    └── app.py           # customtkinter 图形界面（含任务管理和日志重定向）
```

## 1. 环境准备

在开始之前，请确保您的 macOS 系统已安装以下基础组件：

1.  **Python 3.9+**: 建议使用 `3.10` 或以上版本。
2.  **FFmpeg**: 用于音频解码和提取。
    ```bash
    brew install ffmpeg
    ```
3.  **Apple Silicon (M1/M2/M3/M4)**: `mlx-whisper` 需要 Apple 芯片支持。
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
pip install -r requirements.txt
```

`requirements.txt` 包含以下依赖：
- `mlx-whisper` — MLX Whisper 转录引擎
- `numpy` / `torch` — 数值计算和 VAD 模型
- `customtkinter` — GUI 图形界面
- `audio-separator[cpu]` — 人声提取（可选）
- `py2app` — macOS App 打包（可选）

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

### 方式二：GUI 图形界面 (推荐)

本项目提供了易用的图形界面，无需记忆复杂的命令行参数。

1.  **直接启动**：
    ```bash
    python gui_main.py
    ```

2.  **打包为 macOS App (py2app)**：
    
    您可以将项目打包为标准的 `.app` 应用程序，双击即可运行。

    **安装 py2app**:
    ```bash
    pip install py2app
    ```

    **开发模式打包 (Alias - 推荐)**：
    生成的 App 只是一个软链接，体积小，修改代码后无需重新打包，立即生效。
    ```bash
    python setup.py py2app -A
    ```

    启动：在 `dist` 目录中找到 `MlxVadSRT.app` 双击运行。

    **添加到应用程序文件夹**：
    您可以将 `dist/MlxVadSRT.app` 拖入 `/Applications` (应用程序) 文件夹。
    这样您就可以通过 Launchpad (启动台) 或 Spotlight 随时启动它了。

    > **注意**：由于使用 Alias 模式 (`-A`)，生成的 App 只是一个包裹了环境路径的"传送门"。**请不要删除或移动本项目源代码文件夹**，否则 App 将无法运行。

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
| **基础转录 (中文)** | `mlxvad --video demo.mp4 --lang zh` |
| **转录音频 (自动检测)** | `mlxvad --audio record.mp3 --lang auto` |
| **转录 + 嵌入 (生成软字幕视频)** | `mlxvad --video demo.mp4 --embed` |
| **转录 + 翻译 + 嵌入 (一条龙)** | `mlxvad --video demo.mp4 --to zh --embed` |
| **纯翻译 (已有字幕)** | `mlxvad --srt source.srt --to zh` |
| **纯嵌入 (已有字幕)** | `mlxvad --embed --video demo.mp4 --srt source.srt` |
| **翻译 + 嵌入 (已有字幕)** | `mlxvad --embed --video demo.mp4 --srt source.srt --to zh` |
| **去噪后转录 (有 BGM)** | `mlxvad --video movie.mp4 --denoise` |

> **说明**：使用 `--denoise` 时，人声提取的临时文件会存放在系统临时目录（前缀 `mlxvadsrt_vocals_`），任务完成后自动清理。如果任务异常中断，可手动清理该目录。

## 6. 参数说明

- `--audio`: 输入音频文件路径。转录模式下与 `--video`、`--srt` 只能指定其中一个。
- `--video`: 输入视频文件路径。转录模式下与 `--audio`、`--srt` 只能指定其中一个。嵌入模式（`--embed`）下**必须指定**。
- `--srt`: 输入已有 SRT 字幕文件路径。翻译模式下配合 `--to` 使用；嵌入模式（`--embed`）下可选（若未提供，则自动从视频生成）。
- `--embed`: 开启字幕嵌入模式。将生成的或指定的字幕作为软字幕嵌入视频。
    - **配合 `--video` (无 `--srt`)**: 自动执行 [转录 -> (翻译) -> 嵌入] 全流程。
    - **配合 `--video` + `--srt`**: 将指定字幕嵌入视频（若有 `--to` 则先翻译再嵌入）。
    - **输出**: 生成新文件（如 `demo_embedded.mp4`），不覆盖原视频。嵌入成功后自动删除中间 SRT 文件。
- `--lang`: 指定语言 (默认: `auto` 自动检测, 可选: `zh, en, ja, ko, auto`)。仅在转录模式下有效。
- `--to`: 将字幕翻译为指定语言 (默认: 不翻译, 可选: `zh, en, ja, ko`)。转录模式下不能与 `--lang` 相同。
- `--model`: MLX 模型路径或 HF 仓库 (默认: `mlx-community/whisper-large-v3-mlx`)。**注意**：仅支持 `mlx-community/whisper` 系列模型。仅在转录模式下有效。
- `--output`: 输出 SRT 文件名 (默认: 跟随输入文件名，如 `demo.mp4` → `demo.srt`)。配合 `--to` 使用时，显式指定则翻译文件保存到该路径，未指定则自动命名为 `原文件名.目标语言.srt`。
- `--denoise`: 转录前先用 MDX-NET 模型提取人声，去除背景音乐和音效。需额外安装 `audio-separator` 库。适用于电影、电视剧、综艺等有 BGM 的场景。

---

## 7. 高级用法示例

### 🚀 终极用法：转录 + 翻译 + 嵌入 (一条龙)
直接将生肉视频转换为带中文字幕的成品视频：
```bash
mlxvad --video movie.mp4 --lang en --to zh --embed
```
> **流程**：自动转录英文 -> 自动翻译为中文 -> 自动嵌入中文字幕 -> 生成 `movie_embedded.mp4` -> 清理临时字幕文件。

### 场景：仅转录并嵌入 (无需翻译)
适用于同语言字幕生成：
```bash
mlxvad --video lecture.mp4 --embed
```

### 场景：仅翻译现有字幕并嵌入
如果你已经有了英文字幕 `eng.srt`，想翻译成中文并嵌入视频：
```bash
mlxvad --embed --video movie.mp4 --srt eng.srt --to zh
```

### 场景：手动分步处理 (传统模式)
如果你想保留中间文件，或者手动校对字幕后再嵌入，可以分步执行：

1. **转录并翻译**：
   ```bash
   mlxvad --video demo.mp4 --to zh
   # 生成 demo.original.srt 和 demo.zh.srt (保留这两个文件用于校对)
   ```

2. **(可选) 手动修改 demo.zh.srt**...

3. **嵌入校对后的字幕**：
   ```bash
   mlxvad --embed --video demo.mp4 --srt demo.zh.srt
   ```

### 场景：去噪 + 一条龙
对于嘈杂环境的视频：
```bash
mlxvad --video vlog.mp4 --to zh --embed --denoise
```

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

## 9. 配置参数参考

以下参数在 `core/config.py` 中定义，可按需调整：

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `VAD_THRESHOLD` | `0.25` | 语音概率阈值，越低越敏感（最低 `0.1`） |
| `VAD_THRESHOLD_DENOISE` | `0.35` | 去噪后使用的较高阈值（去噪后信号更干净） |
| `VAD_MIN_SILENCE_MS` | `500` | 最短静音时长 (毫秒) |
| `VAD_MIN_SPEECH_MS` | `50` | 最短语音时长 (毫秒) |
| `VAD_SPEECH_PAD_MS` | `300` | 语音片段前后填充 (毫秒) |
| `TRANSLATE_BATCH_SIZE` | `50` | 每批翻译的字幕条数 |
| `TRANSLATE_MAX_WORKERS` | `5` | 翻译并发线程数 |
| `TRANSLATE_MAX_RETRIES` | `5` | 翻译最大重试次数 |
| `TRANSLATE_API_TIMEOUT` | `200` | 翻译请求超时 (秒) |
| `DENOISE_MODEL` | `UVR-MDX-NET-Inst_HQ_3.onnx` | 人声提取使用的 MDX-NET 模型 |

---

## 常见问题
- **首次运行**: 程序会自动从 Hugging Face 下载模型，请保持网络畅通。
- **离线使用**: 在 `main.py` 中取消 `os.environ["HF_HUB_OFFLINE"] = "1"` 的注释，可强制使用本地缓存。
- **翻译超时**: 如果翻译 API 响应较慢，可修改 `core/config.py` 中的 `TRANSLATE_API_TIMEOUT` 常量（默认 200 秒）。
- **台词遗漏**: 如果发现部分语音未被识别到，可降低 `core/config.py` 中的 `VAD_THRESHOLD` 值（默认 0.25，最低 0.1）。
- **去噪后过度过滤**: 如果使用 `--denoise` 后反而丢失台词，可降低 `core/config.py` 中的 `VAD_THRESHOLD_DENOISE` 值（默认 0.35）。
- **`--denoise` 报 ImportError**: 请确保安装了 `audio-separator` 库：`pip install "audio-separator[cpu]"`。
- **`--denoise` 模型下载失败**: 模型缓存在 `~/.cache/audio-separator-models/`，可手动下载后放入该目录。

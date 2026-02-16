"""MlxVadSRT GUI — customtkinter 图形界面"""

import os
import sys
import json
import queue
import threading
import argparse
from tkinter import filedialog, END
from typing import Optional

import customtkinter as ctk

from core.config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS

# 语言选项：显示名 → 代码
LANG_OPTIONS = {
    "自动检测": "auto",
    "简体中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
}

TARGET_LANG_OPTIONS = {
    "不翻译": None,
    "简体中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
}

AUDIO_EXTS = AUDIO_EXTENSIONS
VIDEO_EXTS = VIDEO_EXTENSIONS

# 窗口位置配置文件路径
_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "mlxvadsrt")
_WINDOW_CONFIG = os.path.join(_CONFIG_DIR, "window.json")


# 日志消息类型标记
_MSG_APPEND = 0   # 正常追加
_MSG_CR_LINE = 1  # \r 回车覆盖当前行


class CancelledError(Exception):
    """任务被用户取消"""
    pass


class TextRedirector:
    """将 stdout/stderr 重定向到 queue，处理 \r 进度条覆盖，支持取消检测"""

    def __init__(self, log_queue: queue.Queue, original, cancel_event: threading.Event = None):
        self.queue = log_queue
        self.original = original
        self.cancel_event = cancel_event
        self._buffer = ""

    def write(self, text: str):
        if not text:
            return
        # 检查取消标志
        if self.cancel_event and self.cancel_event.is_set():
            raise CancelledError("任务已取消")
        # 同时输出到原始终端（方便调试）
        if self.original:
            self.original.write(text)

        self._buffer += text

        # 按 \r 和 \n 处理缓冲区
        while self._buffer:
            # 先找 \n（换行）
            nl = self._buffer.find("\n")
            cr = self._buffer.find("\r")

            if nl == -1 and cr == -1:
                # 没有换行也没有回车，暂存
                break

            if nl != -1 and (cr == -1 or nl <= cr):
                # \n 在前：输出到 \n 为止（含换行）
                line = self._buffer[:nl + 1]
                self._buffer = self._buffer[nl + 1:]
                self.queue.put((_MSG_APPEND, line))
            else:
                # \r 在前：表示进度条覆盖
                # 丢弃 \r 之前的内容（已被覆盖）
                self._buffer = self._buffer[cr + 1:]
                # 找下一个 \r 或 \n，提取新行
                next_cr = self._buffer.find("\r")
                next_nl = self._buffer.find("\n")
                if next_nl != -1 and (next_cr == -1 or next_nl <= next_cr):
                    line = self._buffer[:next_nl + 1]
                    self._buffer = self._buffer[next_nl + 1:]
                    self.queue.put((_MSG_CR_LINE, line))
                elif next_cr != -1:
                    line = self._buffer[:next_cr]
                    self._buffer = self._buffer[next_cr:]  # 留着 \r 给下一轮
                    if line:
                        self.queue.put((_MSG_CR_LINE, line))
                else:
                    # 没有结束符，标记为覆盖行但暂存
                    if self._buffer:
                        self.queue.put((_MSG_CR_LINE, self._buffer))
                        self._buffer = ""

    def flush(self):
        # 刷新时把缓冲区剩余内容发出
        if self._buffer:
            self.queue.put((_MSG_APPEND, self._buffer))
            self._buffer = ""
        if self.original:
            self.original.flush()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── 窗口基本设置 ──
        self.title("MlxVadSRT")
        self.minsize(640, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── 恢复窗口位置 ──
        self._load_geometry()

        # ── 关闭时保存位置 ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # 拦截 macOS Cmd+Q
        self.createcommand("::tk::mac::Quit", self._on_close)
        # 窗口移动/缩放时自动保存（防抖）
        self._geo_save_id = None
        self.bind("<Configure>", self._on_configure)

        # ── 状态 ──
        self.log_queue: queue.Queue = queue.Queue()
        self.is_running: bool = False
        self.cancel_event: threading.Event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # ── 构建 UI ──
        self._build_ui()

        # ── 定时轮询日志 ──
        self._poll_queue()

    # ================================================================
    #  窗口位置记忆
    # ================================================================

    def _load_geometry(self):
        """从配置文件恢复窗口位置和大小"""
        try:
            if os.path.exists(_WINDOW_CONFIG):
                with open(_WINDOW_CONFIG, "r") as f:
                    cfg = json.load(f)
                geo = f"{cfg['width']}x{cfg['height']}+{cfg['x']}+{cfg['y']}"
                self.geometry(geo)
                return
        except Exception:
            pass
        # 默认大小
        self.geometry("720x800")

    def _save_geometry(self):
        """保存当前窗口位置和大小到配置文件"""
        try:
            os.makedirs(_CONFIG_DIR, exist_ok=True)
            cfg = {
                "width": self.winfo_width(),
                "height": self.winfo_height(),
                "x": self.winfo_x(),
                "y": self.winfo_y(),
            }
            with open(_WINDOW_CONFIG, "w") as f:
                json.dump(cfg, f)
        except Exception:
            pass

    def _on_close(self):
        """窗口关闭时保存位置并退出"""
        self._save_geometry()
        self.destroy()

    def _on_configure(self, event):
        """窗口移动/缩放时防抖保存位置"""
        if event.widget is not self:
            return
        if self._geo_save_id is not None:
            self.after_cancel(self._geo_save_id)
        self._geo_save_id = self.after(500, self._save_geometry)

    # ================================================================
    #  UI 构建
    # ================================================================

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # 日志区域可伸缩

        # ── 上半部分：设置区 ──
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 8))
        content.grid_columnconfigure(0, weight=1)

        row = 0

        # 标题
        ctk.CTkLabel(
            content, text="MlxVadSRT",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).grid(row=row, column=0, sticky="w")
        row += 1

        ctk.CTkLabel(
            content, text="MLX Whisper + VAD 智能字幕工具",
            font=ctk.CTkFont(size=14), text_color="gray",
        ).grid(row=row, column=0, sticky="w", pady=(0, 16))
        row += 1

        # ── 输入文件 ──
        ctk.CTkLabel(
            content, text="输入文件",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(8, 4))
        row += 1

        input_frame = ctk.CTkFrame(content)
        input_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        input_frame.grid_columnconfigure(0, weight=1)

        self.file_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="选择音频、视频或 SRT 文件...",
            height=36,
        )
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(12, 6), pady=10)

        ctk.CTkButton(
            input_frame, text="浏览", width=72, height=36,
            command=self._browse_file,
        ).grid(row=0, column=1, padx=(0, 12), pady=10)
        row += 1

        # ── 设置 ──
        ctk.CTkLabel(
            content, text="设置",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(8, 4))
        row += 1

        settings_frame = ctk.CTkFrame(content)
        settings_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_frame.grid_columnconfigure(3, weight=1)

        # 第一行：源语言 + 模型
        ctk.CTkLabel(settings_frame, text="源语言").grid(
            row=0, column=0, padx=(12, 6), pady=8, sticky="w")
        self.lang_var = ctk.StringVar(value="自动检测")
        ctk.CTkOptionMenu(
            settings_frame, values=list(LANG_OPTIONS.keys()),
            variable=self.lang_var, width=140,
        ).grid(row=0, column=1, padx=(0, 16), pady=8, sticky="w")

        ctk.CTkLabel(settings_frame, text="模型").grid(
            row=0, column=2, padx=(0, 6), pady=8, sticky="w")
        self.model_entry = ctk.CTkEntry(settings_frame, height=32)
        self.model_entry.insert(0, "mlx-community/whisper-large-v3-mlx")
        self.model_entry.grid(row=0, column=3, padx=(0, 12), pady=8, sticky="ew")

        # 第二行：翻译至 + 选项
        ctk.CTkLabel(settings_frame, text="翻译至").grid(
            row=1, column=0, padx=(12, 6), pady=8, sticky="w")
        self.to_var = ctk.StringVar(value="不翻译")
        ctk.CTkOptionMenu(
            settings_frame, values=list(TARGET_LANG_OPTIONS.keys()),
            variable=self.to_var, width=140,
        ).grid(row=1, column=1, padx=(0, 16), pady=8, sticky="w")

        options_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        options_frame.grid(row=1, column=2, columnspan=2, padx=(0, 12), pady=8, sticky="w")

        self.denoise_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_frame, text="降噪", variable=self.denoise_var,
            checkbox_width=20, checkbox_height=20,
        ).grid(row=0, column=0, padx=(0, 16))

        self.embed_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_frame, text="嵌入字幕", variable=self.embed_var,
            checkbox_width=20, checkbox_height=20,
        ).grid(row=0, column=1)
        row += 1

        # ── 输出路径 ──
        ctk.CTkLabel(
            content, text="输出路径",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(8, 4))
        row += 1

        output_frame = ctk.CTkFrame(content)
        output_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        output_frame.grid_columnconfigure(0, weight=1)

        self.output_entry = ctk.CTkEntry(
            output_frame,
            placeholder_text="留空则自动生成（与输入同名）",
            height=36,
        )
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(12, 6), pady=10)

        ctk.CTkButton(
            output_frame, text="浏览", width=72, height=36,
            command=self._browse_output,
        ).grid(row=0, column=1, padx=(0, 12), pady=10)
        row += 1

        # ── 操作按钮 ──
        self.start_btn = ctk.CTkButton(
            content, text="▶  开始处理", height=46,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._toggle_task,
        )
        self.start_btn.grid(row=row, column=0, sticky="ew", pady=(4, 8))
        row += 1

        # ── 下半部分：日志区域 ──
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 20))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))
        log_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_header, text="日志",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            log_header, text="清除", width=56, height=26,
            fg_color="transparent", border_width=1,
            text_color="gray", hover_color=("gray80", "gray30"),
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Menlo", size=12),
            wrap="word",
            activate_scrollbars=True,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))

    # ================================================================
    #  文件对话框
    # ================================================================

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[
                ("媒体文件", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a *.flac *.srt"),
                ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("音频文件", "*.mp3 *.wav *.m4a *.aac *.flac"),
                ("字幕文件", "*.srt"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="选择输出路径",
            defaultextension=".srt",
            filetypes=[("SRT 字幕", "*.srt"), ("所有文件", "*.*")],
        )
        if path:
            self.output_entry.delete(0, END)
            self.output_entry.insert(0, path)

    # ================================================================
    #  日志
    # ================================================================

    def _log(self, msg: str):
        self.log_queue.put(msg + "\n")

    def _clear_log(self):
        self.log_text.delete("1.0", END)

    def _poll_queue(self):
        """定时从 queue 读取日志并显示，支持 \r 进度条覆盖"""
        try:
            while True:
                item = self.log_queue.get_nowait()

                # 兼容旧格式（纯字符串，来自 _log 方法）
                if isinstance(item, str):
                    self.log_text.insert(END, item)
                    self.log_text.see(END)
                    continue

                msg_type, text = item

                if msg_type == _MSG_CR_LINE:
                    # 删除最后一行内容，替换为新文本
                    self.log_text.delete("end-1c linestart", "end-1c")
                    self.log_text.insert("end-1c", text.rstrip("\n"))
                    if text.endswith("\n"):
                        self.log_text.insert(END, "\n")
                else:
                    self.log_text.insert(END, text)

                self.log_text.see(END)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ================================================================
    #  任务控制
    # ================================================================

    def _toggle_task(self):
        if self.is_running:
            self._cancel_task()
            return
        self._start_task()

    def _cancel_task(self):
        """取消当前运行的任务"""
        self.cancel_event.set()
        self.start_btn.configure(text="⏳  正在取消...", state="disabled")
        self._log("\n⚠️ 正在取消任务...")

    def _start_task(self):
        input_path = self.file_entry.get().strip()

        if not input_path:
            self._log("❌ 错误: 请先选择输入文件。")
            return
        if not os.path.exists(input_path):
            self._log(f"❌ 错误: 文件不存在: {input_path}")
            return

        # 更新状态
        self.is_running = True
        self.cancel_event.clear()
        self.start_btn.configure(
            text="⏹  取消处理", state="normal",
            fg_color="#b33", hover_color="#933",
        )
        self._clear_log()

        # 后台线程执行
        self.worker_thread = threading.Thread(
            target=self._run_task, args=(input_path,), daemon=True
        )
        self.worker_thread.start()

    def _run_task(self, input_path: str):
        """在后台线程中运行核心任务"""
        # 重定向 stdout/stderr（带取消检测）
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TextRedirector(self.log_queue, old_stdout, self.cancel_event)
        sys.stderr = TextRedirector(self.log_queue, old_stderr, self.cancel_event)

        try:
            self._execute_task(input_path)
        except CancelledError:
            self._log("\n⚠️ 任务已取消。")
        except SystemExit:
            # core 模块调用了 sys.exit()，不能让它关闭 GUI
            pass
        except KeyboardInterrupt:
            self._log("\n⚠️ 任务被中断。")
        except Exception as e:
            if self.cancel_event.is_set():
                self._log("\n⚠️ 任务已取消。")
            else:
                self._log(f"\n❌ 任务出错: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.after(0, self._task_finished)

    def _execute_task(self, input_path: str):
        """构建参数并调用 core 模块"""
        ext = os.path.splitext(input_path)[1].lower()

        # 构建与 CLI 一致的 args 对象
        args = argparse.Namespace()
        args.audio = None
        args.video = None
        args.srt = None

        if ext == ".srt":
            args.srt = input_path
        elif ext in VIDEO_EXTS:
            args.video = input_path
        else:
            args.audio = input_path

        args.lang = LANG_OPTIONS.get(self.lang_var.get(), "auto")

        to_display = self.to_var.get()
        args.to = TARGET_LANG_OPTIONS.get(to_display)

        args.model = self.model_entry.get().strip() or "mlx-community/whisper-large-v3-mlx"

        output = self.output_entry.get().strip()
        args.output = output if output else None

        args.denoise = self.denoise_var.get()
        args.embed = self.embed_var.get()

        # ── 参数校验 ──
        if args.srt and not args.to:
            print("❌ 错误: 选择了 SRT 文件时必须指定翻译目标语言。")
            return

        if args.embed and not args.video and not args.srt:
            print("❌ 错误: 嵌入字幕需要视频文件。")
            return

        if not args.srt and args.to and args.to == args.lang:
            print("❌ 错误: 源语言和目标语言不能相同。")
            return

        # ── 翻译配置 ──
        if args.to:
            from core.translate import get_translate_config, check_translate_api
            api_key, base_url, model_name = get_translate_config()
            print("正在检查翻译API可用性...")
            check_translate_api(api_key, base_url, model_name)
            args.translate_config = (api_key, base_url, model_name)
        else:
            args.translate_config = None

        # ── 纯嵌入模式 ──
        if args.embed and args.video and args.srt and not args.to:
            from core.embed import embed_subtitle
            embed_subtitle(args)
            return

        # ── 主任务 ──
        final_srt_path = None
        if args.srt:
            from core.translate import translate_srt_file
            final_srt_path = translate_srt_file(args)
        else:
            from core.transcribe import transcribe_with_vad
            final_srt_path = transcribe_with_vad(args)

        # ── 自动嵌入 ──
        if args.embed and final_srt_path and os.path.exists(final_srt_path):
            print(f"\n--- 正在执行字幕嵌入 (SRT: {os.path.basename(final_srt_path)}) ---")
            args.srt = final_srt_path
            from core.embed import embed_subtitle
            embed_subtitle(args, auto_generated_srt=True)

    def _task_finished(self):
        """任务结束后恢复 UI 状态（在主线程调用）"""
        self.is_running = False
        self.cancel_event.clear()
        self.start_btn.configure(
            text="▶  开始处理", state="normal",
            fg_color=("#3a7ebf", "#1f6aa5"), hover_color=("#325882", "#144870"),
        )


def run():
    """GUI 启动入口"""
    app = App()
    app.mainloop()

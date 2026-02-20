"""MlxVadSRT Web UI — 基于 Gradio"""

import os
import sys
import queue
import threading
import gradio as gr

from core.pipeline import TaskParams, TaskResult, detect_file_type, run_task

# ── 语言选项映射 ────────────────────────────────────────────

LANG_OPTIONS = ["自动检测", "简体中文", "English", "日本語", "한국어"]
LANG_CODE_MAP = {
    "自动检测": "auto",
    "简体中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
}

TARGET_LANG_OPTIONS = ["不翻译", "简体中文", "English", "日本語", "한국어"]
TARGET_LANG_CODE_MAP = {
    "不翻译": None,
    "简体中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
}

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


# ── 日志流式输出辅助 ─────────────────────────────────────────


class _TerminalBuffer:
    """模拟终端行为的缓冲区，正确处理 \\r（进度覆盖）和 \\n"""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._current: str = ""

    def write(self, text: str) -> None:
        for ch in text:
            if ch == "\n":
                self._lines.append(self._current)
                self._current = ""
            elif ch == "\r":
                self._current = ""
            else:
                self._current += ch

    def snapshot(self) -> str:
        """返回当前缓冲区快照"""
        tail = self._lines[-1000:]
        if self._current:
            tail = [*tail, self._current]
        return "\n".join(tail)


class _StreamCapture:
    """临时劫持 stdout/stderr 并将输出发送到队列"""

    def __init__(self, original_stream, log_queue: queue.Queue, cancel_flag: dict) -> None:
        self._original = original_stream
        self._queue = log_queue
        self._cancel = cancel_flag

    def write(self, text: str) -> None:
        if self._cancel.get("is_cancelled"):
            raise KeyboardInterrupt("任务已取消")
        if text:
            self._queue.put(text)
            self._original.write(text)

    def flush(self) -> None:
        self._original.flush()


# ── 核心处理 ─────────────────────────────────────────────


def _build_params(
        file_path: str,
        output_path: str,
        src_lang: str,
        target_lang: str,
        model_name: str,
        denoise: bool,
        embed: bool,
) -> TaskParams:
    """从 Web UI 输入构建 TaskParams"""
    file_path = os.path.expanduser(file_path)
    audio, video, srt = detect_file_type(file_path)
    output = os.path.expanduser(output_path.strip()) if output_path.strip() else None

    return TaskParams(
        audio=audio,
        video=video,
        srt=srt,
        lang=LANG_CODE_MAP.get(src_lang, "auto"),
        to=TARGET_LANG_CODE_MAP.get(target_lang),
        model=model_name.strip() or DEFAULT_MODEL,
        output=output,
        denoise=denoise,
        embed=embed,
    )


def process_file_stream(
        file_path: str,
        output_path: str,
        src_lang: str,
        target_lang: str,
        model_name: str,
        denoise: bool,
        embed: bool,
):
    """以生成器方式运行任务，实时 yield log_text。

    任务在子线程中执行；主线程通过队列轮询日志并 yield 给 Gradio。
    当 Gradio 取消事件时，生成器的 finally 会设置 cancel_flag 终止子线程。
    """
    if not file_path or not file_path.strip():
        yield "❌ 错误: 请输入文件路径。"
        return

    file_path = file_path.strip()
    if not os.path.exists(file_path):
        yield f"❌ 错误: 文件不存在: {file_path}"
        return

    params = _build_params(file_path, output_path, src_lang, target_lang, model_name, denoise, embed)

    log_queue: queue.Queue = queue.Queue()
    cancel_flag = {"is_cancelled": False}
    result_holder: dict = {"result": TaskResult()}

    def _worker() -> None:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamCapture(old_stdout, log_queue, cancel_flag)
        sys.stderr = _StreamCapture(old_stderr, log_queue, cancel_flag)
        try:
            result_holder["result"] = run_task(params)
        except KeyboardInterrupt:
            log_queue.put("\n⚠️ 任务已取消。\n")
        except Exception as e:
            import traceback
            log_queue.put(f"\n❌ 任务出错: {e}\n{traceback.format_exc()}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_queue.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    buf = _TerminalBuffer()

    try:
        while True:
            try:
                chunk = log_queue.get(timeout=0.1)
            except queue.Empty:
                yield buf.snapshot()
                continue

            if chunk is None:
                break
            buf.write(chunk)

            # 批量读取队列中已有的内容
            while not log_queue.empty():
                try:
                    next_chunk = log_queue.get_nowait()
                    if next_chunk is None:
                        chunk = None
                        break
                    buf.write(next_chunk)
                except queue.Empty:
                    break

            yield buf.snapshot()
            if chunk is None:
                break

        # 最终输出：附加结果路径信息
        result = result_holder["result"]
        if result.success and result.output_path:
            buf.write(f"\n✅ 输出文件: {result.output_path}\n")
        yield buf.snapshot()

    finally:
        cancel_flag["is_cancelled"] = True


# ── UI 构建 ──────────────────────────────────────────────


def _create_input_panel() -> tuple:
    """构建左侧输入面板，返回所有输入组件"""
    file_input = gr.Textbox(
        label="输入文件路径",
        placeholder="/path/to/video.mp4 或 audio.wav 或 subtitle.srt",
    )
    output_path = gr.Textbox(
        label="输出路径 (可选)",
        placeholder="留空自动生成；勾选嵌入字幕时为视频路径，否则为 SRT 路径",
    )

    with gr.Group():
        gr.Markdown("### 设置")
        src_lang = gr.Dropdown(choices=LANG_OPTIONS, value="自动检测", label="源语言")
        target_lang = gr.Dropdown(choices=TARGET_LANG_OPTIONS, value="不翻译", label="翻译至")
        model_name = gr.Textbox(value=DEFAULT_MODEL, label="模型")

        with gr.Row():
            denoise = gr.Checkbox(label="降噪 (提取人声去BGM)", value=False)
            embed = gr.Checkbox(label="嵌入字幕", value=False)

    with gr.Row():
        start_btn = gr.Button("▶ 开始处理", variant="primary")
        stop_btn = gr.Button("⏹ 取消任务", variant="stop", visible=False)

    return file_input, output_path, src_lang, target_lang, model_name, denoise, embed, start_btn, stop_btn


# 自动滚动日志框到底部 + 左右等高
_CUSTOM_HEAD = """
<script>
setInterval(() => {
    const ta = document.querySelector('#logs_box textarea');
    if (ta) ta.scrollTop = ta.scrollHeight;
}, 300);
</script>
"""

_CUSTOM_CSS = """
/* 让外层撑满列高度并作为 flex 容器 */
#logs_box {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

/* 匹配 Gradio 的内部 label 容器，让它继续往下伸展 */
#logs_box > label,
#logs_box > label > div {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
    height: 100% !important;
}

/* 核心：让 textarea 吃掉剩余的全部高度 */
#logs_box textarea {
    flex-grow: 1 !important;
    height: 100% !important;
    resize: none !important;
}
"""


def create_ui() -> gr.Blocks:
    """构建 Gradio UI 并绑定事件"""
    with gr.Blocks(title="MlxVadSRT", head=_CUSTOM_HEAD, css=_CUSTOM_CSS) as app:
        gr.Markdown("# MlxVadSRT\n### MLX Whisper + VAD 智能字幕工具 (Web 端)")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                (
                    file_input, output_path, src_lang, target_lang, model_name,
                    denoise, embed, start_btn, stop_btn,
                ) = _create_input_panel()

            with gr.Column(scale=1):
                logs_output = gr.Textbox(
                    label="运行日志",
                    elem_id="logs_box"
                )

        # ── 事件绑定 ──
        # NOTE: cancels 必须引用生成器所在的事件，否则取消不生效。
        # 因此将链条拆开，单独保存生成器事件的引用。

        toggle_start = start_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=None,
            outputs=[start_btn, stop_btn],
            queue=False,
        )

        gen_event = toggle_start.then(
            fn=process_file_stream,
            inputs=[file_input, output_path, src_lang, target_lang, model_name, denoise, embed],
            outputs=[logs_output],
        )

        gen_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=None,
            outputs=[start_btn, stop_btn],
        )

        stop_btn.click(
            fn=None, inputs=None, outputs=None, cancels=[gen_event],
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=None,
            outputs=[start_btn, stop_btn],
            queue=False,
        )

    return app


# ── 入口 ─────────────────────────────────────────────────


def run() -> None:
    app = create_ui()
    app.queue().launch(server_name="127.0.0.1", server_port=8001, inbrowser=True)


if __name__ == "__main__":
    run()

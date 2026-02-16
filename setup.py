"""
py2app 打包配置（别名模式）

使用方法:
    conda run -n mlx python setup.py py2app -A

启动:
    open dist/MlxVadSRT.app
"""

from setuptools import setup

APP = ["gui_main.py"]
DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/MlxVadSRT.icns",
    "plist": {
        "CFBundleName": "MlxVadSRT",
        "CFBundleDisplayName": "MlxVadSRT",
        "CFBundleIdentifier": "com.mlxvadsrt.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "13.0",
        "NSHumanReadableCopyright": "MIT License",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Media File",
                "CFBundleTypeExtensions": [
                    "mp4", "mkv", "mov", "avi", "webm",
                    "mp3", "wav", "m4a", "aac", "flac",
                    "srt",
                ],
                "CFBundleTypeRole": "Viewer",
            }
        ],
    },
    "packages": [
        "core",
        "gui",
        "customtkinter",
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)

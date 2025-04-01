import PyInstaller.__main__
import os
import sys
import platform
import cv2
import numpy
import PyQt6
import matplotlib

# 獲取當前目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 獲取 Python 庫的路徑
cv2_path = os.path.dirname(cv2.__file__)
numpy_path = os.path.dirname(numpy.__file__)
pyqt6_path = os.path.dirname(PyQt6.__file__)
matplotlib_path = os.path.dirname(matplotlib.__file__)

# 根據作業系統選擇路徑分隔符號
separator = ';' if sys.platform == 'win32' else ':'

# 定義圖標路徑（如果有圖標的話）
# icon_path = os.path.join(current_dir, 'icon.ico')  # Windows
# icon_path = os.path.join(current_dir, 'icon.icns')  # macOS

# 定義打包參數
options = [
    'video_processor_ui.py',  # 主程式
    '--name=VideoProcessor',  # 應用程式名稱
    '--onefile',  # 打包成單一檔案
    '--windowed',  # 使用 GUI 模式
    '--clean',  # 清理暫存檔案
    '--noconfirm',  # 不詢問確認
    '--add-data=README.md:.',  # 添加 README 檔案
    '--add-data=entitlements.plist:.',  # 添加權限設定檔案
    
    # 添加必要的隱藏導入
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PyQt6',
    '--hidden-import=PyQt6.QtCore',
    '--hidden-import=PyQt6.QtGui',
    '--hidden-import=PyQt6.QtWidgets',
    '--hidden-import=colorsys',
    '--hidden-import=json',
    '--hidden-import=subprocess',
    '--hidden-import=datetime',
    '--hidden-import=pathlib',
    '--hidden-import=matplotlib',
    '--hidden-import=matplotlib.pyplot',
    '--hidden-import=matplotlib.backends.backend_qt5agg',
    '--hidden-import=matplotlib.backends.backend_agg',
    
    # 添加數據文件
    f'--add-data={cv2_path}/data/*{separator}cv2/data',  # OpenCV 數據文件
    f'--add-data={matplotlib_path}/mpl-data/*{separator}matplotlib/mpl-data',  # Matplotlib 數據文件
    
    # 添加二進制文件
    f'--add-binary={cv2_path}/cv2/*{separator}cv2',  # OpenCV 二進制文件
]

# 根據作業系統添加特定選項
if sys.platform == 'darwin':  # macOS
    # 檢查是否為 Apple Silicon
    is_arm = platform.machine() == 'arm64'
    if is_arm:
        # 如果是 Apple Silicon，只打包 arm64 版本
        options.extend([
            '--target-architecture=arm64',
            '--codesign-identity=-',
            '--osx-entitlements-file=entitlements.plist',
        ])
    else:
        # 如果是 Intel，只打包 x86_64 版本
        options.extend([
            '--target-architecture=x86_64',
            '--codesign-identity=-',
            '--osx-entitlements-file=entitlements.plist',
        ])
elif sys.platform == 'win32':  # Windows
    options.extend([
        # '--icon=' + icon_path,  # 如果有圖標的話
    ])

# 執行打包
PyInstaller.__main__.run(options) 
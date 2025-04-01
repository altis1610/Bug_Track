import os
import sys
import platform
import subprocess

def run_nuitka():
    # 基本命令
    cmd = [
        "python", "-m", "nuitka",
        "--follow-imports",  # 自動追蹤導入
        "--assume-yes-for-downloads",  # 自動下載需要的檔案
        "--remove-output",  # 移除舊的輸出檔案
        "--standalone",  # 創建獨立目錄
        "--show-progress",  # 顯示進度
        "--show-memory",  # 顯示記憶體使用
        "--include-package=cv2",  # 包含 OpenCV
        "--include-package=numpy",  # 包含 NumPy
        "--include-package=matplotlib",  # 包含 Matplotlib
        "--include-data-file=README.md=README.md",  # 包含 README
        "--include-data-file=entitlements.plist=entitlements.plist",  # 包含權限設定
        "--output-dir=dist",  # 輸出目錄
    ]
    
    # 根據作業系統添加特定選項
    if platform.system() == "Windows":
        cmd.extend([
            "--mingw64",  # Windows 使用 mingw64 編譯器
            "--windows-disable-console",  # 禁用控制台視窗
            "--windows-icon-from-ico=icon.ico",  # Windows 圖標
            "--plugin-enable=pyqt6",  # Windows 使用 PyQt6
        ])
    elif platform.system() == "Darwin":  # macOS
        cmd.extend([
            "--macos-create-app-bundle",  # 創建 macOS 應用程式包
            "--macos-disable-console",  # 禁用控制台
            "--macos-app-icon=icon.icns" if os.path.exists("icon.icns") else "",  # 如果有圖標則添加
            "--plugin-enable=pyside6",  # macOS 使用 PySide6
        ])
    
    # 添加主程式
    cmd.append("video_processor_ui.py")
    
    # 移除空字串
    cmd = [x for x in cmd if x]
    
    # 執行命令
    print("開始打包...")
    print("命令:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("打包完成！")

if __name__ == "__main__":
    # 檢查是否已安裝 nuitka
    try:
        import nuitka
    except ImportError:
        print("正在安裝 Nuitka...")
        subprocess.run([sys.executable, "-m", "pip", "install", "nuitka"], check=True)
        print("Nuitka 安裝完成！")
    
    # 檢查是否已安裝 PySide6 (macOS)
    if platform.system() == "Darwin":
        try:
            import PySide6
        except ImportError:
            print("正在安裝 PySide6...")
            subprocess.run([sys.executable, "-m", "pip", "install", "PySide6"], check=True)
            print("PySide6 安裝完成！")
    
    # 執行打包
    run_nuitka() 
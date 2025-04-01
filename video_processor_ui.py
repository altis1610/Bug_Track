import sys
import os
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback
import platform

# 根據作業系統選擇 Qt 綁定
if platform.system() == "Darwin":
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                                 QListWidget, QProgressBar, QSpinBox, QMessageBox,
                                 QFrame, QDoubleSpinBox, QScrollArea, QSizePolicy)
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QIcon, QFont, QPalette, QColor
else:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                                QListWidget, QProgressBar, QSpinBox, QMessageBox,
                                QFrame, QDoubleSpinBox, QScrollArea, QSizePolicy)
    from PyQt6.QtCore import Qt, QThread, Signal, QTimer
    from PyQt6.QtGui import QIcon, QFont, QPalette, QColor

from simple_insect_tracker import process_video

class VideoProcessingThread(QThread):
    progress = Signal(str, int, int)  # video_name, current_frame, total_frames
    finished = Signal(str)  # video_name
    error = Signal(str, str)  # video_name, error_message

    def __init__(self, video_path, output_dir, parameters):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.parameters = parameters
        self.video_name = Path(video_path).stem

    def run(self):
        try:
            def progress_callback(name, progress_info):
                try:
                    current, total = progress_info
                    print(f"Emitting progress: {self.video_name}, {current}/{total}")
                    # 直接在主線程中更新UI
                    self.progress.emit(self.video_name, current, total)
                except Exception as e:
                    print(f"Progress callback error: {str(e)}")
            
            # 處理影片
            process_video(
                video_path=self.video_path,
                output_dir=self.output_dir,
                min_area=self.parameters['min_area'],
                max_area=self.parameters['max_area'],
                min_movement=self.parameters['min_movement'],
                min_duration_seconds=self.parameters['min_duration'],
                buffer_seconds=self.parameters['buffer'],
                display=False,
                progress_callback=progress_callback
            )
            
            self.finished.emit(self.video_name)
        except Exception as e:
            print(f"處理錯誤: {str(e)}")
            self.error.emit(self.video_name, str(e))

class ProgressFrame(QFrame):
    def __init__(self, video_name):
        super().__init__()
        self.video_name = video_name
        self.initUI()
        
    def initUI(self):
        # 設置框架樣式
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
        """)
        
        # 創建垂直布局
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # 檔案名稱標籤
        self.name_label = QLabel(self.video_name)
        self.name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.name_label)
        
        # 進度條容器
        progress_container = QWidget()
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                text-align: center;
                background-color: #F0F0F0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(15)
        progress_layout.addWidget(self.progress_bar)
        
        # 百分比標籤
        self.percentage_label = QLabel("0%")
        self.percentage_label.setFixedWidth(45)
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        progress_layout.addWidget(self.percentage_label)
        
        layout.addWidget(progress_container)
        
        # 設置合適的大小策略
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
    def update_progress(self, current, total):
        try:
            percentage = int((current / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(percentage)
            self.percentage_label.setText(f"{percentage}%")
            self.progress_bar.repaint()
            self.percentage_label.repaint()
            QApplication.processEvents()
        except Exception as e:
            print(f"更新進度時出錯: {str(e)}")
        
    def mark_complete(self):
        try:
            self.progress_bar.setValue(100)
            self.percentage_label.setText("100%")
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #90EE90;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #F0FFF0;
                }
            """)
            self.repaint()
            QApplication.processEvents()
        except Exception as e:
            print(f"標記完成時出錯: {str(e)}")
        
    def mark_error(self, error_msg):
        try:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #F0F0F0;
                }
                QProgressBar::chunk {
                    background-color: #FF6B6B;
                    border-radius: 2px;
                }
            """)
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #FF6B6B;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #FFF0F0;
                }
            """)
            self.percentage_label.setText("錯誤")
            self.repaint()
            QApplication.processEvents()
        except Exception as e:
            print(f"標記錯誤時出錯: {str(e)}")

class VideoProcessorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影片處理器")
        self.setMinimumSize(800, 600)
        
        # 初始化變數
        self.video_paths = []
        self.output_dir = None
        self.processing_threads = {}
        self.progress_frames = {}
        self.max_concurrent = 1  # 預設最大同時處理數量為1
        
        self.init_ui()
        
    def init_ui(self):
        # 主要佈局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左側面板（參數設定）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 參數設定
        parameters_frame = QFrame()
        parameters_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        parameters_layout = QVBoxLayout(parameters_frame)
        
        # 最大同時處理數量
        max_concurrent_layout = QHBoxLayout()
        max_concurrent_layout.addWidget(QLabel("最大同時處理數量:"))
        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(1, 10)
        self.max_concurrent_spin.setValue(1)
        self.max_concurrent_spin.setToolTip("設定同時處理的影片數量上限")
        max_concurrent_layout.addWidget(self.max_concurrent_spin)
        parameters_layout.addLayout(max_concurrent_layout)
        
        # 最小面積
        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("最小面積:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(100)
        min_area_layout.addWidget(self.min_area_spin)
        parameters_layout.addLayout(min_area_layout)
        
        # 最大面積
        max_area_layout = QHBoxLayout()
        max_area_layout.addWidget(QLabel("最大面積:"))
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(0, 100000)
        self.max_area_spin.setValue(0)
        self.max_area_spin.setSpecialValueText("自動")
        max_area_layout.addWidget(self.max_area_spin)
        parameters_layout.addLayout(max_area_layout)
        
        # 最小移動距離
        min_movement_layout = QHBoxLayout()
        min_movement_layout.addWidget(QLabel("最小移動:"))
        self.min_movement_spin = QSpinBox()
        self.min_movement_spin.setRange(1, 1000)
        self.min_movement_spin.setValue(200)
        min_movement_layout.addWidget(self.min_movement_spin)
        parameters_layout.addLayout(min_movement_layout)
        
        # 最短持續時間
        min_duration_layout = QHBoxLayout()
        min_duration_layout.addWidget(QLabel("最短持續時間(秒):"))
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 10.0)
        self.min_duration_spin.setValue(1.0)
        self.min_duration_spin.setSingleStep(0.1)
        min_duration_layout.addWidget(self.min_duration_spin)
        parameters_layout.addLayout(min_duration_layout)
        
        # 緩衝時間
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("緩衝時間(秒):"))
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 5.0)
        self.buffer_spin.setValue(1.0)
        self.buffer_spin.setSingleStep(0.1)
        buffer_layout.addWidget(self.buffer_spin)
        parameters_layout.addLayout(buffer_layout)
        
        left_layout.addWidget(parameters_frame)
        
        # 檔案選擇按鈕
        file_buttons_layout = QHBoxLayout()
        self.add_video_btn = QPushButton("加入影片")
        self.add_video_btn.clicked.connect(self.add_videos)
        self.add_folder_btn = QPushButton("加入資料夾")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.remove_video_btn = QPushButton("移除選中影片")
        self.remove_video_btn.clicked.connect(self.remove_selected_videos)
        file_buttons_layout.addWidget(self.add_video_btn)
        file_buttons_layout.addWidget(self.add_folder_btn)
        file_buttons_layout.addWidget(self.remove_video_btn)
        left_layout.addLayout(file_buttons_layout)
        
        # 影片列表
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 允許多選
        self.video_list.keyPressEvent = self.video_list_key_press  # 覆蓋按鍵事件處理
        left_layout.addWidget(self.video_list)
        
        # 輸出目錄選擇
        output_layout = QHBoxLayout()
        self.output_label = QLabel("輸出目錄: 未選擇")
        self.output_btn = QPushButton("選擇輸出目錄")
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_btn)
        left_layout.addLayout(output_layout)
        
        # 開始處理按鈕
        self.process_btn = QPushButton("開始處理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)
        
        layout.addWidget(left_panel)
        
        # 右側面板（進度顯示）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 進度條容器（可捲動）
        progress_scroll = QScrollArea()
        progress_scroll.setWidgetResizable(True)
        progress_container = QWidget()
        self.progress_layout = QVBoxLayout(progress_container)
        progress_scroll.setWidget(progress_container)
        right_layout.addWidget(progress_scroll)
        
        layout.addWidget(right_panel)
        
        # 設定左右面板的比例
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
    
    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇影片檔案",
            "",
            "影片檔案 (*.mp4 *.avi *.mov *.MOV *.mkv)"
        )
        if files:
            self.video_paths.extend(files)
            self.update_video_list()
    
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇影片資料夾"
        )
        if folder:
            video_extensions = {'.mp4', '.avi', '.mov', '.MOV', '.mkv'}
            for root, _, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix in video_extensions:
                        self.video_paths.append(os.path.join(root, file))
            self.update_video_list()
    
    def update_video_list(self):
        self.video_list.clear()
        for path in self.video_paths:
            self.video_list.addItem(Path(path).name)
        self.update_process_button()
    
    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出目錄"
        )
        if folder:
            self.output_dir = folder
            self.output_label.setText(f"輸出目錄: {folder}")
            self.update_process_button()
    
    def update_process_button(self):
        self.process_btn.setEnabled(
            len(self.video_paths) > 0 and 
            self.output_dir is not None
        )
    
    def start_processing(self):
        # 禁用所有控制項
        self.add_video_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        
        # 清空進度顯示區域
        for i in reversed(range(self.progress_layout.count())): 
            self.progress_layout.itemAt(i).widget().setParent(None)
        
        # 獲取參數
        parameters = {
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value() if self.max_area_spin.value() > 0 else None,
            'min_movement': self.min_movement_spin.value(),
            'min_duration': self.min_duration_spin.value(),
            'buffer': self.buffer_spin.value()
        }
        
        # 設定最大同時處理數量
        self.max_concurrent = self.max_concurrent_spin.value()
        
        # 為每個影片創建進度顯示和處理執行緒
        for video_path in self.video_paths:
            video_name = Path(video_path).name
            
            # 創建進度顯示框架
            progress_frame = ProgressFrame(video_name)
            self.progress_layout.addWidget(progress_frame)
            self.progress_frames[video_name] = progress_frame
            
            # 創建處理執行緒
            thread = VideoProcessingThread(video_path, self.output_dir, parameters)
            
            # 連接信號（使用 lambda 來傳遞參數）
            thread.progress.connect(
                lambda name, curr, tot, frame=progress_frame: 
                frame.update_progress(curr, tot)
            )
            thread.finished.connect(
                lambda name, frame=progress_frame:
                self._thread_finished(frame, thread)
            )
            thread.error.connect(
                lambda name, error, frame=progress_frame:
                frame.mark_error(error)
            )
            
            # 將執行緒加入字典
            self.processing_threads[video_name] = thread
            
            QApplication.processEvents()
        
        # 開始處理第一批影片
        self._start_next_batch()
    
    def _start_next_batch(self):
        """開始處理下一批影片"""
        # 計算當前正在運行的執行緒數量
        active_threads = sum(1 for thread in self.processing_threads.values() if thread.isRunning())
        
        # 如果還有未啟動的執行緒且未達到最大同時處理數量
        for video_name, thread in self.processing_threads.items():
            if not thread.isRunning() and not thread.isFinished() and active_threads < self.max_concurrent:
                thread.start()
                active_threads += 1
    
    def _thread_finished(self, progress_frame, thread):
        """處理執行緒完成時的回調"""
        try:
            progress_frame.mark_complete()
            
            # 檢查是否所有影片都處理完成
            all_finished = all(
                thread.isFinished() for thread in self.processing_threads.values()
            )
            
            if all_finished:
                self._processing_complete()
            else:
                # 啟動下一批處理
                self._start_next_batch()
                
        except Exception as e:
            print(f"處理完成時出錯: {str(e)}")
    
    def _processing_complete(self):
        # 重新啟用所有控制項
        self.add_video_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        
        # 清理
        self.video_paths = []
        self.video_list.clear()
        self.processing_threads.clear()
        self.update_process_button()
    
    def video_list_key_press(self, event):
        # 處理按鍵事件
        if event.key() == Qt.Key.Key_Delete:
            self.remove_selected_videos()
        else:
            # 對於其他按鍵，使用默認處理
            QListWidget.keyPressEvent(self.video_list, event)
    
    def remove_selected_videos(self):
        # 獲取所有選中的項目
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            return
            
        # 獲取選中項目的索引
        selected_indices = [self.video_list.row(item) for item in selected_items]
        selected_indices.sort(reverse=True)  # 從後往前刪除，避免索引變化
        
        # 從列表和路徑中移除選中的影片
        for index in selected_indices:
            self.video_paths.pop(index)
        
        # 更新顯示
        self.update_video_list()

if __name__ == "__main__":
    # 設定日誌
    log_dir = Path.home() / "VideoProcessor_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "video_processor.log"
    
    logging.basicConfig(
        filename=str(log_file),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    app = None
    try:
        logging.info("Starting VideoProcessor application")
        app = QApplication(sys.argv)
        window = VideoProcessorUI()
        window.show()
        logging.info("Application window created and shown")
        sys.exit(app.exec())
    except Exception as e:
        error_msg = f"Error starting application: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        
        # 如果已經有 QApplication 實例，使用它
        if app is None:
            app = QApplication(sys.argv)
        
        # 創建錯誤訊息視窗
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("錯誤")
        error_dialog.setText("應用程式啟動時發生錯誤")
        error_dialog.setDetailedText(error_msg)
        error_dialog.exec()
        
        sys.exit(1) 
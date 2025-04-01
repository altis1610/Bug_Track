import os
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback
from google.colab import files
from google.colab import drive
from IPython.display import display, HTML
import ipywidgets as widgets
from tqdm.notebook import tqdm

def process_video_colab(video_path, output_dir, parameters, progress_callback=None):
    """
    處理單個影片的函數，專門為 Colab 環境優化
    """
    try:
        # 讀取影片
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")

        # 獲取影片資訊
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 創建輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 初始化追蹤器
        tracker = cv2.TrackerCSRT_create()
        tracking = False
        bbox = None
        frame_count = 0
        tracking_start_frame = 0
        tracking_duration = 0
        buffer_frames = int(parameters['buffer'] * fps)

        # 處理每一幀
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)

            # 轉換為灰階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 二值化
            _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
            
            # 形態學操作
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 尋找輪廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 過濾輪廓
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if parameters['min_area'] <= area <= (parameters['max_area'] if parameters['max_area'] else float('inf')):
                    valid_contours.append(contour)
            
            if valid_contours:
                # 找到最大的輪廓
                max_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # 檢查移動距離
                if tracking:
                    prev_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    curr_center = (x + w/2, y + h/2)
                    movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                    (curr_center[1] - prev_center[1])**2)
                    
                    if movement >= parameters['min_movement']:
                        bbox = (x, y, w, h)
                        tracker.init(frame, bbox)
                        tracking_duration = frame_count - tracking_start_frame
                    else:
                        tracking = False
                        bbox = None
                else:
                    bbox = (x, y, w, h)
                    tracker.init(frame, bbox)
                    tracking = True
                    tracking_start_frame = frame_count
            else:
                if tracking:
                    tracking_duration = frame_count - tracking_start_frame
                    if tracking_duration >= parameters['min_duration'] * fps:
                        # 儲存追蹤結果
                        output_file = output_path / f"{Path(video_path).stem}_tracking_{tracking_start_frame}_{frame_count}.json"
                        tracking_data = {
                            'start_frame': tracking_start_frame,
                            'end_frame': frame_count,
                            'duration': tracking_duration / fps,
                            'video_path': str(video_path)
                        }
                        with open(output_file, 'w') as f:
                            json.dump(tracking_data, f, indent=2)
                    tracking = False
                    bbox = None

            # 繪製追蹤框
            if tracking and bbox:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cap.release()
        return True

    except Exception as e:
        print(f"處理影片時發生錯誤: {str(e)}")
        return False

def process_videos_colab():
    """
    在 Colab 環境中處理多個影片的主函數
    """
    # 掛載 Google Drive
    drive.mount('/content/drive')
    
    # 創建參數設定介面
    print("請設定處理參數：")
    min_area = int(input("最小面積 (預設: 100): ") or 100)
    max_area = int(input("最大面積 (預設: 0 表示自動): ") or 0)
    min_movement = int(input("最小移動距離 (預設: 200): ") or 200)
    min_duration = float(input("最短持續時間(秒) (預設: 1.0): ") or 1.0)
    buffer = float(input("緩衝時間(秒) (預設: 1.0): ") or 1.0)
    
    # 設定輸出目錄
    output_dir = input("請輸入輸出目錄路徑 (在 Google Drive 中): ")
    output_dir = os.path.join('/content/drive/MyDrive', output_dir)
    
    # 上傳影片
    print("\n請選擇要處理的影片檔案：")
    uploaded = files.upload()
    
    if not uploaded:
        print("沒有選擇任何檔案！")
        return
    
    # 處理參數
    parameters = {
        'min_area': min_area,
        'max_area': max_area if max_area > 0 else None,
        'min_movement': min_movement,
        'min_duration': min_duration,
        'buffer': buffer
    }
    
    # 處理每個上傳的影片
    for filename, file_content in uploaded.items():
        print(f"\n處理影片: {filename}")
        
        # 創建臨時檔案
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        # 使用 tqdm 顯示進度條
        with tqdm(total=100, desc="處理進度") as pbar:
            def update_progress(current, total):
                progress = int((current / total) * 100)
                pbar.n = progress
                pbar.refresh()
            
            # 處理影片
            success = process_video_colab(
                temp_path,
                output_dir,
                parameters,
                progress_callback=update_progress
            )
        
        # 清理臨時檔案
        os.remove(temp_path)
        
        if success:
            print(f"影片 {filename} 處理完成！")
        else:
            print(f"影片 {filename} 處理失敗！")
    
    print("\n所有影片處理完成！")

if __name__ == "__main__":
    process_videos_colab() 
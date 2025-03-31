import cv2
import numpy as np
import os
import time
from pathlib import Path
import json
import colorsys
import subprocess
from datetime import timedelta

class SimpleDisturbanceTracker:
    """簡化版的擾動追蹤器，專注於準確檢測影像中的移動擾動"""
    
    def __init__(self, min_area=100, max_area=None, min_movement=50, 
                 min_duration_seconds=1.0, fps=30):
        # 基本參數
        self.min_area = min_area
        self.max_area = max_area
        self.min_movement = min_movement
        self.min_duration_seconds = min_duration_seconds
        self.fps = fps
        self.min_frames = int(fps * min_duration_seconds)
        
        # 改善的背景去除器參數
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=20,
            detectShadows=False
        )
        
        # 追蹤狀態
        self.frame_count = 0
        self.disturbances = {}  # {id: {first_frame, last_frame, positions, ...}}
        self.next_id = 1
        
        # 移動檢測參數
        self.min_velocity = 1.0  # 最小速度 (像素/幀)
        self.max_velocity = 50.0  # 最大速度 (像素/幀)
        
        # 雜訊過濾
        self.noise_frames = 3  # 連續出現幀數閾值
        
    def process_frame(self, frame):
        """處理單一影格"""
        if frame is None:
            return None, None
        
        self.frame_count += 1
        vis_frame = frame.copy()
        
        # 1. 背景去除
        fg_mask = self._get_foreground_mask(frame)
        
        # 2. 找出移動物體
        objects = self._detect_objects(fg_mask)
        
        # 3. 更新追蹤
        self._update_tracking(objects)
        
        # 4. 繪製追蹤結果
        self._draw_tracking(vis_frame)
        
        return vis_frame, fg_mask
    
    def _get_foreground_mask(self, frame):
        """獲取並優化前景遮罩"""
        # 套用背景去除
        fg_mask = self.bg_subtractor.apply(frame)
        
        # 形態學操作去除雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return fg_mask
    
    def _detect_objects(self, mask):
        """檢測移動物體"""
        objects = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面積過濾
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            # 計算物體特徵
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 計算物體的其他特徵
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 額外的形狀過濾（昆蟲通常不會是完美的圓形）
            if circularity > 0.9:  # 過濾掉太圓的物體
                continue
            
            objects.append({
                'position': (cx, cy),
                'area': area,
                'contour': contour,
                'circularity': circularity
            })
        
        return objects
    
    def _update_tracking(self, objects):
        """更新擾動追蹤狀態"""
        # 配對現有追蹤和新檢測物體
        matched_objects = set()
        matched_disturbances = set()
        
        # 更新現有追蹤
        for disturbance_id, disturbance in self.disturbances.items():
            if self.frame_count - disturbance['last_frame'] > 10:  # 消失太久的不再追蹤
                continue
                
            if disturbance['positions']:
                last_pos = disturbance['positions'][-1]
                best_match = None
                min_dist = float('inf')
                
                # 尋找最近的物體
                for i, obj in enumerate(objects):
                    if i in matched_objects:
                        continue
                        
                    curr_pos = obj['position']
                    dist = np.sqrt((curr_pos[0] - last_pos[0])**2 + 
                                 (curr_pos[1] - last_pos[1])**2)
                    
                    # 檢查速度是否合理
                    velocity = dist / (self.frame_count - disturbance['last_frame'])
                    if (self.min_velocity <= velocity <= self.max_velocity and 
                        dist < min_dist):
                        min_dist = dist
                        best_match = i
                
                if best_match is not None:
                    matched_obj = objects[best_match]
                    matched_objects.add(best_match)
                    matched_disturbances.add(disturbance_id)
                    
                    # 更新追蹤資訊
                    disturbance['positions'].append(matched_obj['position'])
                    disturbance['areas'].append(matched_obj['area'])
                    disturbance['frames'].append(self.frame_count)
                    disturbance['last_frame'] = self.frame_count
                    disturbance['total_movement'] += min_dist
                    disturbance['missing_frames'] = 0
                else:
                    disturbance['missing_frames'] += 1
        
        # 為未匹配的物體創建新追蹤
        for i, obj in enumerate(objects):
            if i not in matched_objects:
                self.disturbances[self.next_id] = {
                    'id': self.next_id,
                    'first_frame': self.frame_count,
                    'last_frame': self.frame_count,
                    'frames': [self.frame_count],
                    'positions': [obj['position']],
                    'areas': [obj['area']],
                    'total_movement': 0,
                    'missing_frames': 0
                }
                self.next_id += 1
    
    def _draw_tracking(self, frame):
        """在影格上繪製追蹤視覺化"""
        # 檢查當前幀是否有任何有效的擾動
        has_active_disturbance = False
        
        for disturbance_id, disturbance in self.disturbances.items():
            # 檢查是否為當前活動的擾動
            if (self.frame_count - disturbance['last_frame'] <= 30 and 
                disturbance['total_movement'] >= self.min_movement):
                has_active_disturbance = True
                break
        
        # 如果有擾動，在畫面上方寫上 "DETECTED"
        if has_active_disturbance:
            cv2.putText(frame, "DETECTED", 
                       (50, 50),  # 位置在左上角
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5,  # 較大的字體
                       (0, 0, 255),  # 紅色
                       3)  # 較粗的線條
    
    def get_valid_insects(self):
        """獲取符合條件的有效擾動清單"""
        valid_disturbances = []
        
        for disturbance_id, disturbance in self.disturbances.items():
            duration_frames = disturbance['last_frame'] - disturbance['first_frame'] + 1
            duration_seconds = duration_frames / self.fps
            
            # 篩選條件：移動距離、持續時間和連續性
            if (disturbance['total_movement'] >= self.min_movement and 
                duration_frames >= self.min_frames and
                len(disturbance['frames']) >= self.noise_frames):
                
                valid_disturbance = {
                    'id': disturbance_id,
                    'first_frame': disturbance['first_frame'],
                    'last_frame': disturbance['last_frame'],
                    'duration_frames': duration_frames,
                    'duration_seconds': duration_seconds,
                    'total_movement': disturbance['total_movement'],
                    'positions': disturbance['positions'],
                    'frames': disturbance['frames']
                }
                valid_disturbances.append(valid_disturbance)
        
        return valid_disturbances

    def create_timeline_visualization(self, total_duration_seconds):
        """創建擾動時間軸視覺化"""
        # 設定圖片尺寸和邊距
        width = 1500
        height = 200
        margin = 50
        
        # 創建空白圖片
        timeline = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 計算時間軸參數
        axis_start = margin
        axis_end = width - margin
        axis_y = height - margin
        axis_width = axis_end - axis_start
        time_scale = axis_width / total_duration_seconds
        
        # 1. 畫出基本時間軸
        cv2.line(timeline, (axis_start, axis_y), (axis_end, axis_y), (0, 0, 0), 2)
        
        # 2. 畫出時間刻度和標籤
        interval = 5  # 每5秒一個刻度
        for t in range(0, int(total_duration_seconds) + 1, interval):
            x = int(axis_start + t * time_scale)
            # 畫刻度線
            cv2.line(timeline, (x, axis_y), (x, axis_y + 10), (0, 0, 0), 2)
            # 寫時間標籤
            cv2.putText(timeline, f"{t}", 
                       (x - 10, axis_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7,  # 加大字體
                       (0, 0, 0),  # 黑色
                       2)  # 加粗
        
        # 3. 獲取並合併重疊的時間區間
        valid_disturbances = self.get_valid_insects()
        merged_intervals = []
        
        if valid_disturbances:
            intervals = [(d['first_frame'] / self.fps, d['last_frame'] / self.fps, d['id']) 
                        for d in valid_disturbances]
            intervals.sort(key=lambda x: x[0])
            
            current_start, current_end, current_ids = intervals[0]
            current_ids = [current_ids]
            
            for start, end, id_num in intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                    current_ids.append(id_num)
                else:
                    merged_intervals.append((current_start, current_end, current_ids))
                    current_start, current_end = start, end
                    current_ids = [id_num]
            merged_intervals.append((current_start, current_end, current_ids))
        
        # 4. 畫出擾動區間
        bar_height = 20
        bar_y = axis_y - 40  # 放在時間軸上方
        
        for start_time, end_time, ids in merged_intervals:
            # 計算區間的起始和結束位置
            start_x = int(axis_start + start_time * time_scale)
            end_x = int(axis_start + end_time * time_scale)
            
            # 畫藍色區間
            cv2.rectangle(timeline,
                         (start_x, bar_y - bar_height//2),
                         (end_x, bar_y + bar_height//2),
                         (255, 128, 0),  # 橘色
                         -1)  # 填充
            
            # 在區間上方標註時間
            duration = end_time - start_time
            text = f"{start_time:.1f}-{end_time:.1f}s"
            cv2.putText(timeline, text,
                       (start_x, bar_y - bar_height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 0),
                       1)
        
        # 5. 添加標題
        cv2.putText(timeline, "Time (seconds)", 
                   (width//2 - 80, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 0, 0),
                   2)
        
        return timeline


def process_video(video_path, output_dir=None, min_area=100, max_area=None,
                 min_movement=50, min_duration_seconds=1.0, display=True):
    """處理影片並只輸出有檢測到擾動的片段"""
    # 檢查影片路徑
    if not os.path.exists(video_path):
        print(f"錯誤: 找不到影片 {video_path}")
        return None
    
    # 創建輸出目錄
    video_name = Path(video_path).stem
    if output_dir is None:
        output_dir = Path(video_path).parent / f"{video_name}_disturbances"
    else:
        output_dir = Path(output_dir) / f"{video_name}_disturbances"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出目錄: {output_dir}")
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟影片 {video_path}")
        return None
    
    # 獲取影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定最大區域
    if max_area is None:
        max_area = width * height // 20
    
    print(f"影片: {video_path}")
    print(f"解析度: {width}x{height}, FPS: {fps}, 總幀數: {total_frames}")
    print(f"參數: 最小面積={min_area}, 最大面積={max_area}, "
          f"最小移動={min_movement}, 最短持續時間={min_duration_seconds}秒")
    
    # 創建追蹤器
    tracker = SimpleDisturbanceTracker(
        min_area=min_area,
        max_area=max_area,
        min_movement=min_movement,
        min_duration_seconds=min_duration_seconds,
        fps=fps
    )
    
    # 處理影片
    print("\n開始處理影片...")
    start_time = time.time()
    
    # 根據 display 參數選擇進度顯示方式
    if display:
        progress_iter = range(total_frames)
    else:
        from tqdm import tqdm
        progress_iter = tqdm(range(total_frames), desc="Processing", unit="frames")
    
    # 用於記錄要保存的幀
    frames_to_save = []
    frame_indices = []
    
    for _ in progress_iter:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 處理幀
        vis_frame, _ = tracker.process_frame(frame)
        
        # 檢查當前幀是否有活動的擾動
        has_active_disturbance = False
        for disturbance in tracker.disturbances.values():
            if (tracker.frame_count - disturbance['last_frame'] <= 30 and 
                disturbance['total_movement'] >= tracker.min_movement):
                has_active_disturbance = True
                break
        
        # 如果有擾動，保存這一幀
        if has_active_disturbance:
            frames_to_save.append(vis_frame)
            frame_indices.append(tracker.frame_count)
        
        # 顯示處理過程
        if display:
            cv2.imshow('Tracking', vis_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            # 只在 display=True 時顯示詳細進度
            if tracker.frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = tracker.frame_count / elapsed
                remaining = (total_frames - tracker.frame_count) / fps_processing
                print(f"處理進度: {tracker.frame_count}/{total_frames} "
                      f"({tracker.frame_count/total_frames*100:.1f}%) "
                      f"- 處理速度: {fps_processing:.1f} FPS "
                      f"- 剩餘時間: {remaining:.1f} 秒")
    
    # 清理
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # 獲取有效擾動
    valid_insects = tracker.get_valid_insects()
    
    # 合併重疊的時間區間
    if valid_insects:
        intervals = [(d['first_frame'], d['last_frame']) for d in valid_insects]
        intervals.sort(key=lambda x: x[0])
        
        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= current_end + 30:  # 允許30幀的間隔
                current_end = max(current_end, end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intervals.append((current_start, current_end))
        
        # 使用 ffmpeg 切割影片
        for i, (start_frame, end_frame) in enumerate(merged_intervals):
            # 計算時間戳記
            start_time = start_frame / fps
            duration = (end_frame - start_frame) / fps
            
            # 格式化時間戳記為 HH:MM:SS.xxx
            start_timecode = str(timedelta(seconds=start_time))
            duration_timecode = str(timedelta(seconds=duration))
            
            # 設定輸出檔案路徑
            segment_path = output_dir / f"{video_name}_segment_{i+1}.mp4"
            
            # 構建 ffmpeg 命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', start_timecode,
                '-t', duration_timecode,
                '-c:v', 'h264',          # 使用 h264 編碼
                '-preset', 'ultrafast',   # 最快的編碼速度
                '-crf', '23',            # 合理的畫質（0-51，越低越好）
                '-y',                     # 覆寫現有檔案
                str(segment_path)
            ]
            
            # 執行 ffmpeg 命令
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"已保存片段 {i+1}: {segment_path}")
            except subprocess.CalledProcessError as e:
                print(f"處理片段 {i+1} 時發生錯誤: {e}")
                print(f"FFmpeg 錯誤輸出: {e.stderr.decode()}")
    
    print(f"\n處理完成! 總幀數: {tracker.frame_count}")
    print(f"檢測到的有效擾動數量: {len(valid_insects)}")
    print(f"輸出片段數量: {len(merged_intervals)}")
    
    # 更新結果 JSON
    results = {
        'video_path': str(video_path),
        'segments': [
            {
                'path': str(output_dir / f"{video_name}_segment_{i+1}.mp4"),
                'start_frame': start,
                'end_frame': end,
                'start_time': start/fps,
                'end_time': end/fps,
                'duration': (end-start)/fps
            }
            for i, (start, end) in enumerate(merged_intervals)
        ],
        'total_frames': tracker.frame_count,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'parameters': {
            'min_area': min_area,
            'max_area': max_area,
            'min_movement': min_movement,
            'min_duration_seconds': min_duration_seconds
        },
        'disturbances': valid_insects
    }
    
    # 保存結果到JSON
    results_path = output_dir / f"{video_name}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results_path


if __name__ == "__main__":
    import argparse
    
    # 直接設定參數，不使用 argparse
    video_path = "/Users/altis/Downloads/IMG_0319.MOV"  # 輸入的影片檔案路徑
    output_dir = None  # 輸出目錄 (預設為影片所在資料夾)
    min_area = 100  # 最小物體面積 (像素)
    max_area = None  # 最大物體面積 (像素，預設為影像面積的1/20)
    min_movement = 200  # 最小移動距離
    min_duration = 1.0  # 最短持續時間 (秒)
    display = False  # 是否顯示處理過程

    process_video(
        video_path=video_path,
        output_dir=output_dir,
        min_area=min_area,
        max_area=max_area,
        min_movement=min_movement,
        min_duration_seconds=min_duration,
        display=display
    )
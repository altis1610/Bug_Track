import cv2
import numpy as np
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from simple_insect_tracker import SimpleDisturbanceTracker, process_video
import colorsys

class TrajectoryTracker:
    """使用稀疏光流追蹤昆蟲軌跡"""
    
    def __init__(self, min_trajectory_length=15):  # 減少最小軌跡長度
        # 光流參數
        self.feature_params = dict(
            maxCorners=200,  # 增加特徵點數量
            qualityLevel=0.1,  # 降低品質要求
            minDistance=5,  # 減少最小距離
            blockSize=5
        )
        
        self.lk_params = dict(
            winSize=(21, 21),  # 增加窗口大小
            maxLevel=3,  # 增加金字塔層數
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 背景去除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50,  # 減少歷史幀數
            varThreshold=16,  # 降低閾值
            detectShadows=False
        )
        
        # 軌跡參數
        self.min_trajectory_length = min_trajectory_length
        self.trajectories = []
        self.active_trajectories = []
        self.colors = []
        
    def _get_foreground_mask(self, frame):
        """獲取前景遮罩"""
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 擴大前景區域
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        return fg_mask
    
    def process_segment(self, video_path, start_frame, end_frame):
        """處理影片片段並追蹤軌跡"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return None
            
        # 設定起始幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 讀取第一幀
        ret, old_frame = cap.read()
        if not ret:
            print("無法讀取影片幀")
            cap.release()
            return None
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # 初始化追蹤
        self.trajectories = []
        self.active_trajectories = []
        self.colors = []
        
        # 在第一幀中檢測特徵點
        mask = self._get_foreground_mask(old_frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **self.feature_params)
        
        if p0 is not None:
            for point in p0:
                x, y = point[0]
                # 檢查座標是否在影像範圍內
                if (0 <= int(y) < mask.shape[0] and 
                    0 <= int(x) < mask.shape[1] and 
                    mask[int(y), int(x)] > 0):
                    self.active_trajectories.append([point[0].tolist()])
                    self.colors.append(np.random.randint(0, 255, 3).tolist())
        
        frame_idx = start_frame
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = self._get_foreground_mask(frame)
            
            # 如果有活動的軌跡點
            if len(self.active_trajectories) > 0:
                p0 = np.float32([trajectory[-1] for trajectory in self.active_trajectories]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
                
                if p1 is not None:
                    # 獲取好的軌跡點
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # 更新軌跡
                    new_trajectories = []
                    new_colors = []
                    
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        if st[i]:
                            x, y = new.ravel()
                            # 檢查座標是否在影像範圍內
                            if (0 <= int(y) < mask.shape[0] and 
                                0 <= int(x) < mask.shape[1]):  # 移除mask檢查，允許點短暫離開前景
                                # 計算移動距離
                                old_x, old_y = old.ravel()
                                dist = np.sqrt((x - old_x)**2 + (y - old_y)**2)
                                
                                # 如果移動距離合理（不是突然的大跳動）
                                if dist < 30:  # 最大允許的幀間移動距離
                                    self.active_trajectories[i].append([x, y])
                                    new_trajectories.append(self.active_trajectories[i])
                                    new_colors.append(self.colors[i])
                    
                    self.active_trajectories = new_trajectories
                    self.colors = new_colors
            
            # 每隔幾幀檢測新的特徵點
            if frame_idx % 3 == 0:  # 更頻繁地檢測新特徵點
                mask = self._get_foreground_mask(frame)
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                if p0 is not None:
                    for point in p0:
                        x, y = point[0]
                        # 檢查座標是否在影像範圍內
                        if (0 <= int(y) < mask.shape[0] and 
                            0 <= int(x) < mask.shape[1] and 
                            mask[int(y), int(x)] > 0):
                            self.active_trajectories.append([point[0].tolist()])
                            self.colors.append(np.random.randint(0, 255, 3).tolist())
            
            # 更新
            old_gray = frame_gray.copy()
            frame_idx += 1
        
        cap.release()
        
        # 過濾並保存有效軌跡
        valid_trajectories = []
        min_movement = 20  # 最小移動距離閾值
        
        for trajectory in self.active_trajectories:
            if len(trajectory) >= self.min_trajectory_length:
                # 計算總移動距離
                total_movement = 0
                points = np.array(trajectory)
                for i in range(1, len(points)):
                    total_movement += np.linalg.norm(points[i] - points[i-1])
                
                # 計算起點和終點的距離
                start_end_dist = np.linalg.norm(points[-1] - points[0])
                
                # 只保留有足夠移動且不是原地打轉的軌跡
                if total_movement >= min_movement and start_end_dist > 10:
                    valid_trajectories.append(trajectory)
        
        print(f"找到 {len(valid_trajectories)} 條有效軌跡")
        return valid_trajectories
    
    def plot_trajectories(self, trajectories, image_shape, output_path):
        """將軌跡繪製成圖片"""
        # 創建一個白色背景
        background = np.ones(image_shape, dtype=np.uint8) * 255
        
        # 為每個軌跡分配一個獨特的顏色
        colors = []
        for i in range(len(trajectories)):
            # 使用HSV色彩空間來生成均勻分布的顏色
            hue = i / len(trajectories)
            color = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
            colors.append(color)
        
        # 繪製所有軌跡
        for trajectory, color in zip(trajectories, colors):
            # 只處理足夠長的軌跡
            if len(trajectory) >= self.min_trajectory_length:
                # 轉換為整數座標
                points = np.array(trajectory, dtype=np.int32)
                
                # 繪製軌跡線
                for i in range(1, len(points)):
                    cv2.line(background,
                            tuple(points[i-1]),
                            tuple(points[i]),
                            color,
                            2)
                
                # 標記起點和終點
                cv2.circle(background, tuple(points[0]), 5, (0, 0, 255), -1)  # 紅色起點
                cv2.circle(background, tuple(points[-1]), 5, (0, 255, 0), -1)  # 綠色終點
                
                # 添加軌跡編號
                cv2.putText(background,
                          f"T{len(colors)}",
                          tuple(points[0]),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 0, 0),
                          1)
        
        # 保存圖片
        cv2.imwrite(str(output_path), background)
        
        return background

def process_video_with_trajectories(video_path, output_dir=None, **kwargs):
    """處理影片並產生軌跡圖"""
    print("第一步：使用 SimpleDisturbanceTracker 檢測擾動區間...")
    
    # 首先使用 SimpleDisturbanceTracker 處理影片
    disturbance_tracker = SimpleDisturbanceTracker(
        min_area=kwargs.get('min_area', 100),
        max_area=kwargs.get('max_area', None),
        min_movement=kwargs.get('min_movement', 50),
        min_duration_seconds=kwargs.get('min_duration_seconds', 1.0)
    )
    
    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return None
    
    # 獲取影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定輸出目錄
    video_name = Path(video_path).stem
    if output_dir is None:
        output_dir = Path(video_path).parent / f"{video_name}_analysis"
    else:
        output_dir = Path(output_dir) / f"{video_name}_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"處理影片: {video_path}")
    print(f"輸出目錄: {output_dir}")
    
    # 處理每一幀
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 處理幀
        disturbance_tracker.process_frame(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"已處理 {frame_count}/{total_frames} 幀")
    
    cap.release()
    
    # 獲取有效的擾動區間
    valid_disturbances = disturbance_tracker.get_valid_insects()
    
    if not valid_disturbances:
        print("未檢測到有效的擾動")
        return None
    
    # 合併重疊的擾動區間
    print("\n合併重疊的擾動區間...")
    
    # 按開始幀排序
    valid_disturbances.sort(key=lambda x: x['first_frame'])
    
    merged_disturbances = []
    current = valid_disturbances[0]
    
    for next_dist in valid_disturbances[1:]:
        # 檢查是否重疊（允許30幀的間隔）
        if next_dist['first_frame'] <= current['last_frame'] + 30:
            # 合併區間
            current['last_frame'] = max(current['last_frame'], next_dist['last_frame'])
            # 合併其他資訊
            current['positions'].extend(next_dist['positions'])
            current['frames'].extend(next_dist['frames'])
            if 'total_movement' in current and 'total_movement' in next_dist:
                current['total_movement'] += next_dist['total_movement']
        else:
            merged_disturbances.append(current)
            current = next_dist
    
    # 添加最後一個區間
    merged_disturbances.append(current)
    
    print(f"合併前區間數量: {len(valid_disturbances)}")
    print(f"合併後區間數量: {len(merged_disturbances)}")
    
    # 更新有效擾動列表
    valid_disturbances = merged_disturbances
    
    print(f"\n檢測到 {len(valid_disturbances)} 個有效擾動區間（合併後）")
    
    # 第二步：對每個擾動區間進行軌跡追蹤和影片切割
    print("\n第二步：處理每個擾動區間...")
    
    # 創建軌跡追蹤器
    trajectory_tracker = TrajectoryTracker(min_trajectory_length=15)
    
    # 處理每個擾動區間
    for i, disturbance in enumerate(valid_disturbances):
        print(f"\n處理擾動區間 {i+1}/{len(valid_disturbances)}")
        print(f"幀範圍: {disturbance['first_frame']} - {disturbance['last_frame']}")
        
        # 1. 切割影片
        segment_path = output_dir / f"segment_{i+1}.mp4"
        start_time = disturbance['first_frame'] / fps
        duration = (disturbance['last_frame'] - disturbance['first_frame']) / fps
        
        # 使用 ffmpeg 切割影片
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'h264',
            '-preset', 'ultrafast',
            '-y',
            str(segment_path)
        ]
        
        try:
            import subprocess
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"已保存影片片段: {segment_path}")
        except subprocess.CalledProcessError as e:
            print(f"切割影片時發生錯誤: {e}")
            continue
        
        # 2. 生成軌跡圖
        trajectories = trajectory_tracker.process_segment(
            video_path,
            disturbance['first_frame'],
            disturbance['last_frame']
        )
        
        if trajectories:
            trajectory_path = output_dir / f"segment_{i+1}_trajectories.png"
            trajectory_tracker.plot_trajectories(trajectories, (height, width, 3), trajectory_path)
            print(f"已保存軌跡圖: {trajectory_path}")
    
    print("\n處理完成！")

if __name__ == "__main__":
    video_path = "/Users/altis/Downloads/IMG_0319.MOV"
    process_video_with_trajectories(
        video_path=video_path,
        output_dir=None,
        min_area=100,
        max_area=None,
        min_movement=200,
        min_duration_seconds=1.0
    ) 
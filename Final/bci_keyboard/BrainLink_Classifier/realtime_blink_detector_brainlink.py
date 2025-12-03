import numpy as np
from collections import deque

class RealTimeBlinkDetector:
    def __init__(self, fs=512, threshold=-180):
        """
        fs: 採樣率 (NeuroSky 通常是 512Hz)
        threshold: 觸發閾值。
                   觀察您的圖片，雜訊約在 0~-100 之間，眨眼會下衝至 -200 以下。
                   因此預設設為 -160 (負值)。
        """
        self.fs = fs
        self.threshold_low = threshold       
        
        # 0.1秒的滑動視窗 (足以平滑掉高頻雜訊，但保留眨眼特徵)
        self.window_len = int(self.fs / 32)
        # 每次推進 0.02秒 計算一次
        self.step_size = int(0.02 * self.fs)
        
        # --- 即時處理用的變數 (State) ---
        self.buffer = deque(maxlen=self.window_len)
        self.step_counter = 0
        
        # DC Blocker (去除直流偏移，讓訊號以 0 為中心)
        self.running_mean = 0.0
        self.alpha = 0.002

        # 狀態變數
        self.is_locked = False   # 是否正在"眨眼訊號中" (防止同一次眨眼重複觸發)
        self.current_peak = -99999.0

        self.current_output = 0  # 當前的輸出 (1=偵測到眨眼, 0=無)
        self.debug_avg = 0.0     # 除錯用：當前的平均值

    def update(self, sample):
        """
        輸入單一個採樣點 (raw value, integer or float)，
        回傳當前的偵測狀態 (1 = Blink Detected, 0 = None)
        """
        # 1. DC Offset Removal (這步很重要，確保訊號以 0 為基準)
        if self.running_mean == 0.0:
            self.running_mean = sample
        else:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * sample
            
        clean_sample = sample - self.running_mean
        
        # 2. 放入 Buffer
        self.buffer.append(clean_sample)
        self.step_counter += 1

        # 預設輸出為 0 (除非觸發的那一瞬間)
        self.current_output = 0

        # 3. 檢查是否達到處理步長
        if len(self.buffer) == self.window_len and self.step_counter >= self.step_size:
            
            self.step_counter = 0 # 重置計數器
            
            # 計算視窗平均值 (平滑化，抗噪)
            avg_val = np.mean(self.buffer)
            self.debug_avg = avg_val

            # === 核心邏輯修改區 ===
            
            if not self.is_locked:
                # [觸發條件]：如果平均值 "低於" 負的閾值 (向下衝)
                # 例如：訊號衝到 -250，低於閾值 -200 -> 觸發
                if avg_val < self.threshold_low:
                    self.is_locked = True
                    self.current_output = 1  # 輸出眨眼訊號
                    # print(f"Blink Detected! Level: {avg_val:.2f}") # 除錯用
            
            else:
                # [重置條件]：訊號回升
                # 為了避免在閾值附近抖動，我們設一個 Hysteresis (遲滯區間)
                # 例如：必須回升到 (-200 + 50 = -150) 以上才算結束
                reset_level = self.threshold_low + 70
                
                if avg_val > reset_level:
                    self.is_locked = False
        return self.current_output
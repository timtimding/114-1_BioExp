import numpy as np
from collections import deque

class RealTimeBlinkDetector:
    def __init__(self, fs=500, threshold=80):
        self.fs = fs
        self.threshold_high = threshold       
        # self.reset_drop = 150                 # 重置下降幅度 (Peak - 150)
        
        # 0.1-second sliding window to eliminate jitters
        self.window_len = int(0.1 * self.fs)
        # Step length = 0.05s
        self.step_size = int(0.05 * self.fs)
        
        # --- 即時處理用的變數 (State) ---
        self.buffer = deque(maxlen=self.window_len) # Buffer for sliding window
        self.step_counter = 0                       # Counter
        
        # DC Blocker
        self.running_mean = 0.0
        self.alpha = 0.002  # Refresh rate of average value

        # State variables
        self.is_locked = False
        self.current_peak = -99999.0
        
        # 輸出狀態保持 (為了讓輸出訊號是連續的方波，而不是只有觸發瞬間是1)
        self.current_output = 0
        self.debug_avg = 0.0

    def update(self, sample):
        """
        輸入單一個採樣點 (float)，回傳當前的偵測狀態 (0 or 1)
        """
        # 1. DC Offset Removal
        # 使用指數移動平均 (Exponential Moving Average) 來模擬 batch mean
        if self.running_mean == 0.0:
            self.running_mean = sample
        else:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * sample
            
        clean_sample = sample - self.running_mean
        
        # 2. Insert buffer
        self.buffer.append(clean_sample)
        self.step_counter += 1

        # 3. 檢查是否達到處理步長 (模擬 step_size)
        # 只有當緩衝區滿了，且計數器達到 step_size 時才運算
        if len(self.buffer) == self.window_len and self.step_counter >= self.step_size:
            
            self.step_counter = 0 # 重置計數器
            
            # 計算視窗平均
            avg_val = np.mean(self.buffer)
            self.debug_avg = avg_val

            if not self.is_locked:
                # Wait to be triggered
                if avg_val > self.threshold_high:
                    self.is_locked = True
                    self.current_peak = avg_val
                    self.current_output = 1
            else:
                self.current_output = 0

                # Triggered -> find peak
                # if avg_val > self.current_peak:
                #     self.current_peak = avg_val
                # reset_threshold = self.current_peak - self.reset_drop
                
                # Here we back to when the amplitude is lower than
                # the threshold to mark the end of a blink.
                if avg_val < self.threshold_high - 30:
                    self.is_locked = False
        
        return self.current_output
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from collections import deque
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy

# === Import blink detector ===
try:
    from realtime_blink_detector import RealTimeBlinkDetector
except ImportError:
    print("File 'realtime_blink_detector.py' not found, check if it's in /src.")
    sys.exit(1)

# === BCI extract features ===
def get_band_power(epoch, fs, bands):
    f, p = signal.welch(epoch, fs, nperseg=len(epoch), window='hann')
    res = f[1]-f[0]; powers = {b: simpson(p[(f>=l)&(f<=h)], dx=res) for b, (l, h) in bands.items()}
    tot = sum(powers.values()); 
    if tot == 0: return ({b:0 for b in bands}, np.array([0]))
    return ({b:v/tot for b,v in powers.items()}, p)

def extract_features_single(segment, fs, bands, f_list):
    EPS=1e-10
    sos=signal.butter(4,[4,30],btype='bandpass',fs=fs,output='sos')
    
    # Hjorth (Filtered)
    seg_c = signal.sosfilt(sos, segment)
    d1=np.diff(seg_c); d2=np.diff(d1); v0=np.var(seg_c); vd1=np.var(d1); vd2=np.var(d2)
    mob=np.sqrt(vd1/(v0+EPS)); comp=(np.sqrt(vd2/(vd1+EPS)))/(mob+EPS)
    
    # Spectral (Raw)
    rel, psd = get_band_power(segment, fs, bands)
    ent = entropy(psd/(np.sum(psd)+EPS))
    
    t, b, a = rel.get('Theta',0), rel.get('Beta',0), rel.get('Alpha',0)
    vec = []
    for name in f_list:
        if name == 'Theta': vec.append(np.log10(t+EPS))
        elif name == 'Beta': vec.append(np.log10(b+EPS))
        elif name == 'AB_Ratio': vec.append(np.log10(a/(b+EPS)))
        elif name == 'TB_Ratio': vec.append(np.log10(t/(b+EPS)))
        elif name == 'A_sum_ABT': vec.append(np.log10(a/(a+b+t+EPS)))
        elif name == 'Spec_Entropy': vec.append(ent)
        elif name == 'Hjorth_Activity': vec.append(np.log10(v0+EPS))
        elif name == 'Hjorth_Mobility': vec.append(mob)
        elif name == 'Hjorth_Complexity': vec.append(comp)
        else: vec.append(0.0)
    return np.array([vec])

# 3. BCI engine (Focus/relax classifier)
class BCIEngine:
    def __init__(self, model_path):
        print(f"Loading BCI Model: {model_path}...")
        try:
            pkg = joblib.load(model_path)
            self.model = pkg['model']
            self.scaler = pkg['scaler']
            self.threshold = pkg['threshold']
            self.cfg = pkg['config']
        except Exception as e:
            raise FileNotFoundError(f"Fail to load model: {e}")

        self.fs = self.cfg['fs']
        self.window_len = int(self.cfg['window'] * self.fs) # 2.5s
        self.buffer = deque(maxlen=self.window_len)
        
        # Debounce
        self.history = deque(maxlen=5)
        self.last_stable_state = 0

    def update(self, chunk):
        """Enter a chunk to get result of prediction"""
        self.buffer.extend(chunk)
        
        if len(self.buffer) < self.window_len:
            return None # Buffer not filled
            
        segment = np.array(self.buffer)
        segment = segment - np.median(segment) # eliminate DC
        
        # Feature extraction and data normalization
        feats = extract_features_single(segment, self.fs, self.cfg['bands'], self.cfg['features'])
        feats_scaled = self.scaler.transform(feats)
        
        # SVM prediction
        if hasattr(self.model, "predict_proba"):
            score = self.model.predict_proba(feats_scaled)[0, 1]
        else:
            dist = self.model.decision_function(feats_scaled)[0]
            score = 1 / (1 + np.exp(-dist))

        raw_pred = 1 if score > self.threshold else 0
        
        # Vote to debounce
        self.history.append(raw_pred)
        final_pred = 1 if np.mean(self.history) > 0.5 else 0
        self.last_stable_state = final_pred
        
        return final_pred

# 4. Integrated system
class IntegratedSystem:
    def __init__(self, model_path, blink_threshold=80, fs=500):
        self.fs = fs
        
        # === Model 1: RealTimeBlinkDetector ===
        self.blink_detector = RealTimeBlinkDetector(fs=fs, threshold=blink_threshold)
        self.blink_detector.reset_drop = 150
        
        # === Model 2: BCI engine ===
        self.bci_engine = BCIEngine(model_path)
        
        # 系統參數
        self.bci_update_interval = int(0.25 * fs) # 每 0.25s 更新一次 BCI
        self.bci_accumulator = []
        
        # 凍結機制 (Freeze Mechanism)
        self.freeze_timer = 0
        self.FREEZE_DURATION = int(0.5 * fs) # 偵測到眨眼後，凍結 BCI 輸出 0.5 秒
        self.last_output = 0 # 預設為放鬆

    def process_sample(self, sample):
        """
        輸入: 單一個採樣點 (float)
        輸出: (blink_moving_avg, blink_state, bci_state)
        """
        # 1. 執行眨眼偵測 (每點都做)
        blink_state = self.blink_detector.update(sample)
        blink_ma = self.blink_detector.debug_avg # 取得模組內部的移動平均值 (用於畫圖)
        
        # 2. 如果發現眨眼 (Rising Edge)，啟動凍結計時器
        if blink_state == 1:
            self.freeze_timer = self.FREEZE_DURATION
            
        # 3. 倒數計時
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
            
        # 4. 累積數據給 BCI
        self.bci_accumulator.append(sample)
        
        # 5. 檢查是否該執行 BCI (每 0.25s)
        current_bci = self.last_output
        
        if len(self.bci_accumulator) >= self.bci_update_interval:
            
            if self.freeze_timer == 0:
                # === 正常模式：執行預測 ===
                pred = self.bci_engine.update(self.bci_accumulator)
                if pred is not None:
                    self.last_output = pred
                    current_bci = pred
            else:
                # === 凍結模式：維持原判 ===
                # (不呼叫 update，節省算力，並避免雜訊汙染 Buffer 導致後續誤判)
                # 但這裡為了讓 BCI 的 Buffer 保持推進 (Slide)，我們還是要 update，只是忽略結果
                # 或者，更簡單的做法：直接忽略這次計算
                # 這裡選擇：僅推進 Buffer 但不採納結果
                self.bci_engine.buffer.extend(self.bci_accumulator) 
                current_bci = self.last_output
            
            # 清空累積器
            self.bci_accumulator = []
            
        return blink_ma, blink_state, current_bci

# ==========================================
# 5. 主程式：載入檔案、模擬與繪圖
# ==========================================
def main():
    # 設定路徑
    MODEL_PATH = 'bci_system_v1.pkl'
    
    TEST_FILE = "blink_data/blink_3.txt" 
    
    if not os.path.exists(MODEL_PATH):
        print("No model found, please train first")
        return

    # 載入資料
    if os.path.exists(TEST_FILE):
        print(f"Reading file: {TEST_FILE}")
        raw_data = np.loadtxt(TEST_FILE)
        # raw_data = raw_data[:500*30] # 只取前 30 秒測試
    else:
        print("No file detected, use simulated data...")
        t = np.linspace(0, 10, 5000)
        raw_data = np.sin(2*np.pi*10*t) * 20 + np.random.normal(0, 5, 5000)
        raw_data[2000:2200] += 300 # 模擬一個大眨眼

    # 初始化系統
    system = IntegratedSystem(MODEL_PATH, blink_threshold=80)
    
    # 紀錄變數 (用於繪圖)
    log_blink_ma = []
    log_blink_st = []
    log_bci_st = []
    
    print(f"🚀 開始處理 {len(raw_data)} 個採樣點...")
    
    # === 模擬串流迴圈 ===
    for sample in raw_data:
        b_ma, b_st, bci_st = system.process_sample(sample)
        
        log_blink_ma.append(b_ma)
        log_blink_st.append(b_st)
        log_bci_st.append(bci_st)
        
    print("✅ 處理完成，正在繪圖...")

    # === 繪圖 ===
    t = np.arange(len(raw_data)) / 500
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 圖 1: 原始訊號 + 眨眼移動平均
    # 為了方便觀察，原始訊號扣掉平均值
    axes[0].plot(t, raw_data - np.mean(raw_data), color='#CCCCCC', label='Raw EEG', lw=0.8)
    axes[0].plot(t, log_blink_ma, color='orange', label='Blink Detector MA', lw=1.5)
    axes[0].axhline(system.blink_detector.threshold_high, color='red', linestyle='--', label='Threshold')
    axes[0].set_title('Raw Signal & Blink Detector Internal State')
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Amplitude (uV)')
    
    # 圖 2: 眨眼判讀 (0/1)
    axes[1].fill_between(t, log_blink_st, color='red', alpha=0.3, step='post')
    axes[1].step(t, log_blink_st, color='red', label='Blink Detected')
    axes[1].set_title('Blink Output')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.1, 1.1)
    
    # 圖 3: BCI 專注判讀 (0/1) + 凍結區間標示
    axes[2].fill_between(t, log_bci_st, color='green', alpha=0.3, step='post')
    axes[2].step(t, log_bci_st, color='green', label='Focus State (BCI)')
    
    # 標示 "凍結區間" (只要有眨眼的地方，BCI 應該是水平直線)
    blink_mask = np.array(log_blink_st) > 0
    # 這裡簡單用眨眼發生當下標示，實際凍結時間會比這更長 (延後 2秒)
    # 為了視覺化清楚，我們畫出 "潛在影響區"
    axes[2].fill_between(t, 0, 1, where=blink_mask, color='gray', alpha=0.2, transform=axes[2].get_xaxis_transform(), label='Blink Occurred')

    axes[2].set_title('BCI Focus Output (with Freeze Mechanism)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Focus')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
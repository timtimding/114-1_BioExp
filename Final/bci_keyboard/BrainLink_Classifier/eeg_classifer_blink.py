import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from collections import deque
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy
import time

MODEL_PATH = 'bci_system_v1.pkl'
TEST_FILE = "blink_data/blink_3.txt"

try:
    from realtime_blink_detector_brainlink import RealTimeBlinkDetector     # BrainLink
    # from realtime_blink_detector import RealTimeBlinkDetector               # BIOPAC
except ImportError:
    print("File 'realtime_blink_detector_brainlink.py' not found, check if it's in /src.")
    sys.exit(1)

# Blink counter
class BlinkSequenceDetector:
    def __init__(self, timeout=0.8, max_count=3, cooldown=2.0):
        self.timeout = timeout          # 超過 0.8秒 沒眨眼視為序列結束
        self.maxcount = max_count       # 單一序列中最多連續眨眼次數
        self.cooldown_dura = cooldown   # 這次眨眼序列結束後要等的秒數
        
        self.counter = 0                # 當前累積次數
        self.last_blink_time = 0        # 上一次眨眼時間
        self.prev_state = 0             # 邊緣偵測用
        self.cooldown_timer = 0         # 等他歸零才會開始下一次序列
    
    def update(self, blink_state):
        """
        輸入: blink_state (0 or 1)
        輸出: final_result (0=無結果, 1=單擊確認, 2=雙擊確認...)
        """
        current_time = time.time()
        output = 0
        
        # 0. 檢查是否在冷卻中
        if self.cooldown_timer > 0:
            if current_time < self.cooldown_timer:
                self.prev_state = blink_state # 重要：更新狀態以防冷卻結束瞬間誤判
                return 0 
            else:
                self.cooldown_timer = 0 # 冷卻結束
                
        # 1. 偵測 Rising Edge (0 -> 1)
        # 只有在訊號剛變為 1 的那個瞬間才計數
        if blink_state == 1 and self.prev_state == 0:
            self.counter += 1
            self.last_blink_time = current_time
            # print(f"Blink count: {self.counter}") 

            # 檢查是否達到最大次數 (例如 3)
            if self.counter >= self.maxcount:
                output = self.counter
                self.counter = 0
                self.cooldown_timer = current_time + self.cooldown_dura
                # print(f"Max count reached! Cooldown for {self.cooldown_dura}s")

        # 2. 檢查超時 (Timeout)
        # 如果當前沒有新的眨眼 (或是訊號持續為1但沒觸發新計數)，且距離上次已超時
        elif self.counter > 0 and (current_time - self.last_blink_time > self.timeout):
            # 只有當訊號已經回到 0 或者是維持狀態時才結算
            # 這裡簡單判斷時間即可，因為如果正在眨眼(state=1)，counter不會增加，時間差也不會重置
            output = self.counter
            self.counter = 0

        self.prev_state = blink_state
        return output

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

# 4. Integrated system (Blink detector & EEG-based Classifier)
class IntegratedSystem:
    def __init__(self, model_path, blink_threshold=-180, fs=512, timeout=0.8, max_count=3, cooldown=2.0):
        self.fs = fs
        
        # === Model 1: RealTimeBlinkDetector ===
        self.blink_detector = RealTimeBlinkDetector(fs=fs, threshold=blink_threshold)   # Brainlink
        # self.blink_detector = RealTimeBlinkDetector(fs=500, threshold=80)               # BIOPAC
        # self.blink_detector.reset_drop = 150
        
        # === Model 2: BCI engine ===
        self.bci_engine = BCIEngine(model_path)
        
        # === Model 3: Blink counter ===
        self.seq_detector = BlinkSequenceDetector(timeout=timeout, max_count=max_count, cooldown=cooldown) # 0.8秒內沒連點算結束


        # System parameters
        self.bci_update_interval = int(0.25 * fs) # Update BCI result every 0.25 second
        self.bci_accumulator = []
        
        # Freeze Mechanism
        self.freeze_timer = 0
        self.FREEZE_DURATION = int(0.5 * fs) # BCI output is disable within 0.5s after blink is detected
        self.last_output = 0

    def process_sample(self, sample):
        """
        Input: single data point
        Output: (blink_moving_avg, blink_state, bci_state)
        """
        # 1. Blink detection
        blink_state = self.blink_detector.update(sample)
        blink_ma = self.blink_detector.debug_avg # Moving average for plotting result
        blink_seq_count = self.seq_detector.update(blink_state)

        # 2. Start "freezing" if the start of a blink is detected
        if blink_state == 1:
            self.freeze_timer = self.FREEZE_DURATION
            
        # 3. Freezing counter
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
            
        # 4. Accumulate data for BCI
        self.bci_accumulator.append(sample)
        
        # 5. Run BCI or not (every 0.25s)
        current_bci = self.last_output
        
        if len(self.bci_accumulator) >= self.bci_update_interval:
            # print("buffer loaded!")   # Check if the buffer updates correctly
            if self.freeze_timer == 0:
                # === 正常模式：執行預測 ===
                pred = self.bci_engine.update(self.bci_accumulator)
                if pred is not None:
                    self.last_output = pred
                    current_bci = pred
            else:
                # === 凍結模式：維持原判 ===
                # 這裡選擇：繼續塞東西到 Buffer 裡，但不理會 BCI classifier 的輸出結果
                self.bci_engine.buffer.extend(self.bci_accumulator)
                current_bci = self.last_output
            
            # Empty buffer
            self.bci_accumulator = []
            
        return blink_ma, blink_state, current_bci, blink_seq_count

# Main (For module testing with external waveform files)
def main():
    if not os.path.exists(MODEL_PATH):
        print("No model found, please train first")
        return

    # Load data
    if os.path.exists(TEST_FILE):
        print(f"Reading file: {TEST_FILE}")
        raw_data = np.loadtxt(TEST_FILE)
    else:
        # Generate simulated data if no file loaded.
        print("No file detected, use simulated data...")
        t = np.linspace(0, 10, 5000)
        raw_data = np.sin(2*np.pi*10*t) * 20 + np.random.normal(0, 5, 5000)
        raw_data[2000:2200] += 300

    # Initialization
    system = IntegratedSystem(MODEL_PATH, blink_threshold=80)
    
    # Variables for plotting
    log_blink_ma = []
    log_blink_st = []
    log_bci_st = []
    
    print(f"Start processing {len(raw_data)} samples...")
    
    # Simulate data stream
    for sample in raw_data:
        b_ma, b_st, bci_st = system.process_sample(sample)
        
        log_blink_ma.append(b_ma)
        log_blink_st.append(b_st)
        log_bci_st.append(bci_st)
        
    print("Simulation done! Plotting the result...")

    # Plot results
    t = np.arange(len(raw_data)) / 500
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Fig 1: Raw data & moving average
    axes[0].plot(t, raw_data - np.mean(raw_data), color='#CCCCCC', label='Raw EEG', lw=0.8)
    axes[0].plot(t, log_blink_ma, color='orange', label='Blink Detector MA', lw=1.5)
    axes[0].axhline(system.blink_detector.threshold_high, color='red', linestyle='--', label='Threshold')
    axes[0].set_title('Raw Signal & Blink Detector Internal State')
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Amplitude (uV)')
    
    # Fig 2: Blink detection result
    axes[1].fill_between(t, log_blink_st, color='red', alpha=0.3, step='post')
    axes[1].step(t, log_blink_st, color='red', label='Blink Detected')
    axes[1].set_title('Blink Output')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.1, 1.1)
    
    # Fig 3: BCI Classifier output (0/1) & Segments interrupted by blinks
    axes[2].fill_between(t, log_bci_st, color='green', alpha=0.3, step='post')
    axes[2].step(t, log_bci_st, color='green', label='Focus State (BCI)')
    
    # 標示 "凍結區間" (只要有眨眼的地方，BCI 應該是水平直線)
    blink_mask = np.array(log_blink_st) > 0
    # 這裡簡單用眨眼發生當下標示，實際凍結時間會比這更長 (延後 0.5秒)
    # 為了視覺化清楚，我們畫出 "潛在影響區"
    axes[2].fill_between(t, 0, 1, where=blink_mask, color='gray', alpha=0.2, transform=axes[2].get_xaxis_transform(), label='Blink Occurred')

    axes[2].set_title('BCI Focus Output (with Freeze Mechanism)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Focus')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("Classifier&Blink.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
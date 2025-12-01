import numpy as np
import os
import glob
import joblib
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# === 1. 設定檔 (與您測試效果最好的設定一致) ===
class Config:
    DATASET_PATH = "bci_dataset_113-3"
    
    BANDS = {'Delta': [1, 5], 'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [12, 30], 'Gamma': [30, 45]}
    
    FEATURE_LIST = [
        'Theta', 'Beta', 'AB_Ratio', 'TB_Ratio', 'A_sum_ABT',
        'Spec_Entropy', 'Hjorth_Mobility', 'Hjorth_Complexity'
    ]
    
    KEEP_RANGES = [
        (20, 40), (60, 80), (100, 120), (140, 160), 
        (180, 200), (220, 240), (260, 280), 
        (300, 320), (340, 360), (380, 400)
    ]
    
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 2.5
    OVERLAP_RATIO = 0.9
    
    # SVM 參數 (基於您提供的設定)
    SVM_C = 1.0
    SVM_KERNEL = 'poly'
    SVM_DEGREE = 4
    SVM_COEF0 = 1.0
    SVM_GAMMA = 'scale'
    SVM_PROBABILITY = True
    SVM_CLASS_WEIGHT = {0: 1.0, 1: 1.3} # 稍微加重專注的權重
    # !可能要改成{0: 1.0, 1: 1.3}, 避免將放鬆錯判成專注
    RANDOM_STATE = 42

# === 2. 訊號處理函數群 (必須與推論時完全一致) ===
def preprocess_signal(data):
    x = np.array(data, dtype=float)
    x = x - np.median(x)
    return x

def get_band_power(epoch, fs, bands):
    f, p = signal.welch(epoch, fs, nperseg=len(epoch), window='hann')
    res = f[1]-f[0]; powers = {b: simpson(p[(f>=l)&(f<=h)], dx=res) for b, (l, h) in bands.items()}
    tot = sum(powers.values()); 
    if tot == 0: return ({b:0 for b in bands}, np.array([0]))
    return ({b:v/tot for b,v in powers.items()}, p)

def extract_features(segments, fs, bands, f_list):
    EPS=1e-10; feats=[]; sos=signal.butter(4,[4,30],btype='bandpass',fs=fs,output='sos')
    
    for seg in segments:
        # 1. Hjorth (Filtered)
        seg_c = signal.sosfilt(sos, seg)
        d1=np.diff(seg_c); d2=np.diff(d1); v0=np.var(seg_c); vd1=np.var(d1); vd2=np.var(d2)
        mob=np.sqrt(vd1/(v0+EPS)); comp=(np.sqrt(vd2/(vd1+EPS)))/(mob+EPS)
        
        # 2. Spectral (Raw)
        rel, psd = get_band_power(seg, fs, bands)
        ent = entropy(psd/(np.sum(psd)+EPS))
        
        # 3. Mapping
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
        feats.append(vec)
    return np.array(feats)

# === 3. Load data and train ===
def load_and_prepare_data():
    all_f = []; all_l = []
    folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    print("Reading data...")
    
    for folder in folders:
        try:
            r = preprocess_signal(np.loadtxt(os.path.join(folder, "1.txt")))
            f = preprocess_signal(np.loadtxt(os.path.join(folder, "2.txt")))
        except: continue
        
        # 使用安全切分提取數據
        seg_len = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
        step = int(seg_len * (1 - Config.OVERLAP_RATIO))
        
        def process(raw, label):
            for start, end in Config.KEEP_RANGES:
                idx_s, idx_e = int(start*Config.SAMPLING_RATE), int(end*Config.SAMPLING_RATE)
                if idx_s >= len(raw): break
                block = raw[idx_s : min(idx_e, len(raw))]
                curr = 0
                while curr + seg_len <= len(block):
                    segment = block[curr : curr + seg_len]
                    feats = extract_features([segment], Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
                    all_f.append(feats[0])
                    all_l.append(label)
                    curr += step

        process(r, 0)
        process(f, 1)
        
    return np.array(all_f), np.array(all_l)

def main():
    X, y = load_and_prepare_data()
    if len(X) == 0: print("No data"); return
    
    print(f"Start training (Sample amount: {len(X)})...")
    
    # 1. 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. 訓練 SVM
    clf = SVC(
        C=Config.SVM_C, kernel=Config.SVM_KERNEL, degree=Config.SVM_DEGREE,
        coef0=Config.SVM_COEF0, gamma=Config.SVM_GAMMA,
        probability=True, class_weight=Config.SVM_CLASS_WEIGHT,
        random_state=Config.RANDOM_STATE
    )
    clf.fit(X_scaled, y)
    
    # 3. 找最佳閾值 (Self-Check)
    prob = clf.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thr = roc_curve(y, prob)
    best_thr = thr[np.argmax(tpr - fpr)]
    print(f"Best threshold found: {best_thr:.4f}")
    
    # 4. 存檔
    package = {
        'model': clf,
        'scaler': scaler,
        'threshold': best_thr,
        'config': {
            'fs': Config.SAMPLING_RATE,
            'window': Config.SEGMENT_LENGTH,
            'features': Config.FEATURE_LIST,
            'bands': Config.BANDS
        }
    }
    joblib.dump(package, 'bci_system_v1.pkl')
    print("Model saved to 'bci_system_v1.pkl'")

if __name__ == "__main__":
    main()
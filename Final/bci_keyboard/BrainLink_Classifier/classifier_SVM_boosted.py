"""
Brain-Computer Interface SVM Classifier (Full Version)
Features:
1. Model: Support Vector Machine (RBF Kernel)
2. Features: Hybrid (Time-Domain + Frequency-Domain) with 4-30Hz filtering
3. Safety: Safe Segmentation (KEEP_RANGES)
4. Output: Exact match with CNN-LSTM version (F1 Score + Plots)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve
import os
import glob
import warnings
# import random
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy

warnings.filterwarnings('ignore')

class Config:
    DATASET_PATH = "bci_dataset_113-3"

    BANDS = {
        'Delta': [1, 5],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [12, 30],
        'Gamma': [30, 45]
    }
    
    # === 特徵列表 (FP1-FP2 優化版) ===
    FEATURE_LIST = [
        'Theta',            # 前額葉專注指標
        'Beta',             # 警覺指標
        'AB_Ratio',         # Alpha/Beta
        'TB_Ratio',         # Theta/Beta (ADHD/專注常用)
        'A_sum_ABT',        # 歸一化 Alpha
        'Spec_Entropy',     # 頻譜熵
        # 'Hjorth_Activity',
        'Hjorth_Mobility',  # 時域頻率
        'Hjorth_Complexity' # 時域頻寬
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST)
    
    # === 安全切分時間段 ===
    KEEP_RANGES = [
        (20, 40), (60, 80), (100, 120), (140, 160), 
        (180, 200), (220, 240), (260, 280), 
        (300, 320), (340, 360), (380, 400)
    ]
    
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 2.5
    OVERLAP_RATIO = 0.9
    
    # === SVM Hyperparameters ===
    SVM_C = 1.0                    # 懲罰係數 (可調: 0.1, 1, 10)
    # SVM_KERNEL = 'rbf'           # 核心函數
    SVM_KERNEL = 'poly'            # 核心函數
    SVM_DEGREE = 4                 # 多項式次數
    SVM_COEF0 = 1.0                # 偏差值
    SVM_GAMMA = 'scale'      
    SVM_PROBABILITY = True         # 必須為 True 以支援後處理

    # SVM_CLASS_WEIGHT = 'balanced'  # 自動平衡權重
    SVM_CLASS_WEIGHT = {0: 1.0, 1: 1.3} # 專注部分懲罰係數較大
    
    RANDOM_STATE = 42

# === Preprocessing ===
def preprocess_signal(data):
    x = np.array(data, dtype=float)
    x = x - np.median(x)
    return x

def load_eeg_data(subject_path):
    relax_file = os.path.join(subject_path, "1.txt")
    focus_file = os.path.join(subject_path, "2.txt")
    try:
        relax_data = np.loadtxt(relax_file)
        focus_data = np.loadtxt(focus_file)
        relax_data = preprocess_signal(relax_data)
        focus_data = preprocess_signal(focus_data)
        return relax_data, focus_data
    except Exception as e:
        print(f"Error loading {subject_path}: {e}")
        return None, None

# === Safe Segmentation ===
def segment_data_safely(raw_data, fs, seg_len_sec, overlap_ratio, keep_ranges):
    segments = []
    seg_points = int(seg_len_sec * fs)
    step = int(seg_points * (1 - overlap_ratio))
    
    for (start_sec, end_sec) in keep_ranges:
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        if start_idx >= len(raw_data): break
        real_end = min(end_idx, len(raw_data))
        block_data = raw_data[start_idx:real_end]
        
        curr = 0
        while curr + seg_points <= len(block_data):
            seg = block_data[curr : curr + seg_points]
            segments.append(seg)
            curr += step
    return np.array(segments)

# === Features (含濾波) ===
def get_band_power(epoch_data, fs, bands):
    freqs, psd = signal.welch(epoch_data, fs, nperseg=len(epoch_data), window='hann')
    freq_res = freqs[1] - freqs[0]; abs_powers = {}
    for band_name, (low_hz, high_hz) in bands.items():
        idx_band = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
        if len(idx_band) == 0: abs_powers[band_name] = 0; continue
        power = simpson(psd[idx_band], dx=freq_res)
        abs_powers[band_name] = power
    total_power = sum(abs_powers.values())
    if total_power == 0: return {band: 0 for band in bands}, np.array([0])
    rel_powers = {band: power / total_power for band, power in abs_powers.items()}
    return rel_powers, psd

def extract_features(segments, fs, bands, feature_list):
    EPSILON = 1e-10; all_features = []
    
    # 建立 4-30Hz 濾波器 (用於 Hjorth)
    sos = signal.butter(4, [4, 30], btype='bandpass', fs=fs, output='sos')
    
    for segment in segments:
        # 1. 計算 Hjorth (使用濾波後數據)
        segment_clean = signal.sosfilt(sos, segment)
        d1 = np.diff(segment_clean); d2 = np.diff(d1)
        var_0 = np.var(segment_clean); var_d1 = np.var(d1); var_d2 = np.var(d2)
        hjorth_mobility = np.sqrt(var_d1 / (var_0 + EPSILON))
        hjorth_complexity = (np.sqrt(var_d2 / (var_d1 + EPSILON))) / (hjorth_mobility + EPSILON)
        
        # 2. 計算頻域 (使用原始數據算 PSD)
        rel_powers, psd = get_band_power(segment, fs, bands)
        psd_norm = psd / (np.sum(psd) + EPSILON); spec_entropy = entropy(psd_norm)
        delta = rel_powers.get('Delta', 0); theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0); beta  = rel_powers.get('Beta', 0)
        
        vec = []
        for f in feature_list:
            if f == 'Theta': vec.append(np.log10(theta))
            elif f == 'Beta': vec.append(np.log10(beta))
            elif f == 'Alpha': vec.append(np.log10(alpha))
            elif f == 'AB_Ratio': vec.append(np.log10(alpha / (beta + EPSILON)))
            elif f == 'TB_Ratio': vec.append(np.log10(theta / (beta + EPSILON))) # 新增
            elif f == 'A_sum_ABT': vec.append(np.log10(alpha / (alpha + beta + theta + EPSILON)))
            elif f == 'Spec_Entropy': vec.append(spec_entropy)
            elif f == 'Hjorth_Activity': vec.append(var_0)
            elif f == 'Hjorth_Mobility': vec.append(hjorth_mobility)
            elif f == 'Hjorth_Complexity': vec.append(hjorth_complexity)
            else: vec.append(0.0)
        all_features.append(vec)
    return np.array(all_features)

# === SVM Classifier Wrapper ===
class EnhancedBCIClassifier:
    def __init__(self):
        # 使用 SVC
        self.model = SVC(
            C=Config.SVM_C,
            kernel=Config.SVM_KERNEL,
            gamma=Config.SVM_GAMMA,

            degree=Config.SVM_DEGREE,
            coef0=Config.SVM_COEF0,

            probability=Config.SVM_PROBABILITY,
            class_weight=Config.SVM_CLASS_WEIGHT,
            random_state=Config.RANDOM_STATE,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.threshold_ = 0.5
        self.smooth_k = 5
        self.vote_k = 7

    def fit(self, X, y):
        # 1. 正規化
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. 打亂數據 (Shuffle)
        idx = np.arange(len(X_scaled))
        np.random.shuffle(idx)
        
        # 3. 訓練 SVM
        self.model.fit(X_scaled[idx], y[idx])
        
        # 4. 閾值調整 (Threshold Tuning)
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.14, random_state=Config.RANDOM_STATE)
            (train_idx, val_idx) = next(sss.split(X_scaled, y))
            val_proba = self.model.predict_proba(X_scaled[val_idx])[:, 1]
            fpr, tpr, thr = roc_curve(y[val_idx], val_proba)
            best = np.argmax(tpr - fpr)
            self.threshold_ = thr[best] if np.isfinite(thr[best]) else 0.5
        except Exception: self.threshold_ = 0.5
        return self

    def _majority_vote(self, y_pred: np.ndarray, k: int) -> np.ndarray:
        if k <= 1: return y_pred
        if k % 2 == 0: k += 1
        pad = k // 2; y_pad = np.pad(y_pred.astype(int), (pad, pad), mode='edge')
        votes = np.convolve(y_pad, np.ones(k, dtype=int), mode='valid')
        return (votes > (k // 2)).astype(int)

    def _postprocess_predict(self, X_selected: np.ndarray) -> np.ndarray:
        # 獲取機率
        proba = self.model.predict_proba(X_selected)[:, 1]
        
        # 平滑機率
        k_s = int(getattr(self, "smooth_k", 1) or 1)
        if k_s > 1:
            kernel = np.ones(k_s, dtype=float) / float(k_s)
            proba = np.convolve(proba, kernel, mode='same')
            
        # 閾值判斷
        pred = (proba > self.threshold_).astype(int)
        
        # 多數決投票
        k_v = int(getattr(self, "vote_k", 1) or 1)
        return self._majority_vote(pred, k_v)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self._postprocess_predict(X_scaled)       

    def get_loss_curve(self):
        # SVM 沒有 loss curve
        return []

# === Main Flow (2D Data) ===
def load_all_safe_data_2d():
    all_f = []; all_l = []; all_s = []
    folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    print(f"Loading Data (SVM)... Ranges: {Config.KEEP_RANGES}")
    
    for folder in folders:
        sid = os.path.basename(folder)
        d1, d2 = load_eeg_data(folder)
        if d1 is None: continue
        
        # 安全切分
        s1 = segment_data_safely(d1, Config.SAMPLING_RATE, Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO, Config.KEEP_RANGES)
        s2 = segment_data_safely(d2, Config.SAMPLING_RATE, Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO, Config.KEEP_RANGES)
        
        if len(s1)==0: continue
        
        # 提取特徵 (回傳 2D 陣列)
        f1 = extract_features(s1, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        f2 = extract_features(s2, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        
        all_f.append(np.vstack([f1, f2]))
        all_l.append(np.hstack([np.zeros(len(f1)), np.ones(len(f2))]))
        all_s.extend([sid]*len(np.hstack([np.zeros(len(f1)), np.ones(len(f2))])))
        
    if not all_f: return None, None, None
    return np.vstack(all_f), np.hstack(all_l), np.array(all_s)

def run_safe_loso_svm():
    print("Starting Safe LOSO (SVM)...")
    X_seg, y_seg, subjects = load_all_safe_data_2d()
    if X_seg is None: return None
    
    unique_subjects = sorted(list(set(subjects)))
    # random.shuffle(unique_subjects) # 打亂驗證順序
    
    # Results containers
    accs = []
    cms = []
    losses = []
    
    for test_sub in unique_subjects:
        train_mask = (subjects != test_sub); test_mask = (subjects == test_sub)
        X_train, y_train = X_seg[train_mask], y_seg[train_mask]
        X_test, y_test = X_seg[test_mask], y_seg[test_mask]
        
        if len(X_train)==0: continue
        
        clf = EnhancedBCIClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        accs.append(acc)
        cms.append(cm)
        losses.append(clf.get_loss_curve())
        
        print(f"{test_sub}: {acc:.3f}")
        
    return {'accuracies': accs, 'confusion_matrices': cms, 'loss_curves': losses}

def plot_results(results):
    if results is None: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'BCI Classifier (SVM) - LOSO Results', fontsize=16)
    
    # 1. Accuracy
    axes[0].bar(range(len(results['accuracies'])), results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.6 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_xlabel('Subject Index')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Confusion Matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus'],
                yticklabels=['Relax', 'Focus'],
                ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # 3. Loss (N/A)
    axes[2].text(0.5, 0.5, 'Loss curve not available for SVM', transform=axes[2].transAxes, ha='center')
    axes[2].set_title('Training Loss Curves')
    
    plt.tight_layout(); plt.savefig('bci_safe_results_svm.png', dpi=300, bbox_inches='tight'); plt.show()

def main():
    print("BCI EEG Classification (SVM Full) - Relaxation vs Concentration"); print("=" * 70)
    np.random.seed(Config.RANDOM_STATE)
    
    results = run_safe_loso_svm()
    if results is None: print("Validation failed!"); return
    
    mean_acc = np.mean(results['accuracies']); std_acc = np.std(results['accuracies'])
    print(f"\nOverall Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_recall = total_cm[0, 0] / np.sum(total_cm[0, :])
        focus_recall = total_cm[1, 1] / np.sum(total_cm[1, :])
        relax_prec = total_cm[0, 0] / np.sum(total_cm[:, 0])
        focus_prec = total_cm[1, 1] / np.sum(total_cm[:, 1])
        relax_f1 = 2 * (relax_prec * relax_recall) / (relax_prec + relax_recall) if (relax_prec + relax_recall) > 0 else 0
        focus_f1 = 2 * (focus_prec * focus_recall) / (focus_prec + focus_recall) if (focus_prec + focus_recall) > 0 else 0

    print(f"\nRelax Class:")
    print(f"  - Accuracy (Recall): {relax_recall:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_prec:.3f}")
    print(f"  - F1 Score: {relax_f1:.3f}")

    print(f"\nConcentration Class:")
    print(f"  - Accuracy (Recall): {focus_recall:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {focus_prec:.3f}")
    print(f"  - F1 Score: {focus_f1:.3f}")
    
    plot_results(results)
    print(f"\nResults saved to 'bci_results_svm_2-5_0.9_poly.png'")

if __name__ == "__main__":
    main()
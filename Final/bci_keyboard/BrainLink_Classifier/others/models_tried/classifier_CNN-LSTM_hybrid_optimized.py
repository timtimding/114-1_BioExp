"""
Brain-Computer Interface CNN-LSTM Classifier
For EEG signal relaxation/focus state classification
Architecture: Conv1D -> MaxPooling -> LSTM -> Dense
"""

"""
Why is this version said to be "optimized"?
 - When preprocessing the data, we simply remove
   the resting segments (0s~20s, 40s~60s, ......), 
   the rest of data will be placed together. The window
   we use is 2.5s, and some data cross the unsuccessive
   part (e.g. 19s~21.5s in trimmed data maps to 39s~40s and
   60~61.5s in original data). This will directly include
   some unwanted noise. Thus in this version we view each 
   20s-segment as independent part.

 - Each 10*35 = 350 segments will be given an unique ID
   and treated as independent parts.
   
"""

"""
Some bad datasets (S03, S19, S22 and S27) will be removed
in further training process. However, it seems to converge
slower in previous tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import glob
import warnings
import random
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy

# === Keras / TensorFlow ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === GPU Memory Setting ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

warnings.filterwarnings('ignore')

class Config:
    DATASET_PATH = "bci_dataset_113-2"

    BANDS = {
        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [13, 30],
        'Gamma': [30, 45]
    }
    
    FEATURE_LIST = [
        # 'Raw',
        'AB_Ratio',
        'A_sum_ABT',
        'Spec_Entropy', 
        'Hjorth_Mobility',
        'Hjorth_Complexity',
        'Hjorth_Activity'
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST)
    
    # === 定義要保留的時間段 (秒) ===
    # 0-20(休), 20-40(測), 40-60(休), 60-80(測)...
    KEEP_RANGES = [
        (20, 40),
        (60, 80),
        (100, 120),
        (140, 160),
        (180, 200),
        (220, 240),
        (260, 280),
        (300, 320),
        (340, 360),
        (380, 400)
    ]
    
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 2.5
    OVERLAP_RATIO = 0.9

    # === CNN-LSTM Hyperparameters ===
    # 
    # 1. 序列長度
    SEQUENCE_TIMESTEPS = 8      # !(可調) 模型一次看幾個窗口 (記憶長度)
    
    # 2. CNN 部分 (特徵提取)
    CNN_FILTERS = 32             # !(可調) 卷積核數量 (提取多少種局部特徵)
    CNN_KERNEL_SIZE = 3          # !(可調) 卷積核大小 (一次看 3 個時間步)
    CNN_POOL_SIZE = 1            # !(可調) 池化大小 (將序列長度縮小幾倍，例如 15 -> 7)
                                 # 這能減輕 LSTM 的負擔並提取顯著特徵
    
    # 3. LSTM 部分 (時序記憶)
    LSTM1_UNITS = 64              # !(可調) LSTM 記憶單元數量
    LSTM2_UNITS = 32              # !(可調) LSTM 記憶單元數量
    #LSTM_RECURRENT_DROPOUT = 0.5 # !(可調) 防止 LSTM 過擬合
    LSTM1_RECURRENT_DROPOUT = 0.0 # !0.0 for CuDNN Speed
    LSTM2_RECURRENT_DROPOUT = 0.0 # !0.0 for CuDNN Speed
    
    # 4. 全連接層
    DENSE_UNITS = 32             # !(可調)
    DROPOUT = 0.5                # !(可調)
    
    # Training parameters
    LEARNING_RATE = 0.0005       # !(可調) 學習率 (建議比純 CNN 稍低)
    EPOCHS = 100                 # !(可調)
    BATCH_SIZE = 16              # !(可調)
    EARLY_STOP_PATIENCE = 15     # !(可調)
    VALIDATION_FRACTION = 0.1    # !(可調)

    # Feature selection
    FEATURE_SELECTION = False          
    N_FEATURES_SELECT = N_FEATURES_TOTAL 
    
    # Other settings
    RANDOM_STATE = 42

# === Preprocessing ===
def preprocess_signal(data):
    # 基本去噪
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

# === 關鍵修改 2: 安全切分函數 (取代舊的 create_segments) ===
def segment_data_safely(raw_data, fs, seg_len_sec, overlap_ratio, keep_ranges, subject_id):
    """
    只在 keep_ranges 定義的區塊內進行滑動窗口切分。
    這保證了不會跨越斷層，也不會包含休息時間。
    """
    segments = []
    group_ids = []
    
    seg_points = int(seg_len_sec * fs)
    step = int(seg_points * (1 - overlap_ratio))
    
    for i, (start_sec, end_sec) in enumerate(keep_ranges):
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        
        # 邊界檢查
        if start_idx >= len(raw_data): break
        real_end = min(end_idx, len(raw_data))
        
        # 提取該區塊
        block_data = raw_data[start_idx:real_end]
        
        # 在區塊內滑動窗口
        curr = 0
        while curr + seg_points <= len(block_data):
            seg = block_data[curr : curr + seg_points]
            segments.append(seg)

            group_ids.append(f"{subject_id}_B{i}")

            curr += step
            
    return np.array(segments), np.array(group_ids)

# === Calculate band power ===
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
    
    # Filter (4-30Hz) to eliminate muscle-related signals
    sos = signal.butter(4, [4, 30], btype='bandpass', fs=fs, output='sos')
    
    for segment in segments:
        # ==========================================
        # 1. Calcultate Hjorth parameters with filtered data
        # ==========================================
        segment_clean = signal.sosfilt(sos, segment)
        
        d1 = np.diff(segment_clean)
        d2 = np.diff(d1)
        
        var_0 = np.var(segment_clean)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        # Calculate the variance
        hjorth_mobility = np.sqrt(var_d1 / (var_0 + EPSILON))
        hjorth_complexity = (np.sqrt(var_d2 / (var_d1 + EPSILON))) / (hjorth_mobility + EPSILON)
        
        # ==========================================
        # 2. 計算頻帶功率 & 熵 (通常使用 *原始* 數據)
        # ==========================================
        # 使用原始 segment，因為 get_band_power 內部是用 Welch 方法，
        # 它能正確處理頻譜，且保留 Delta/Gamma 用於可能的其他特徵分析
        rel_powers, psd = get_band_power(segment, fs, bands)
        
        psd_norm = psd / (np.sum(psd) + EPSILON)
        spec_entropy = entropy(psd_norm)
        
        delta = rel_powers.get('Delta', 0); theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0); beta  = rel_powers.get('Beta', 0)
        gamma = rel_powers.get('Gamma', 0)
        
        # ==========================================
        # 3. 組合特徵向量
        # ==========================================
        feature_vector = []
        for feature_name in feature_list:
            # if feature_name == 'Raw': feature_vector.append(rel_powers)
            if feature_name == 'Delta': feature_vector.append(delta)
            elif feature_name == 'Theta': feature_vector.append(theta)
            elif feature_name == 'Alpha': feature_vector.append(alpha)
            elif feature_name == 'Beta': feature_vector.append(beta)
            elif feature_name == 'A_sum_ABT': den = alpha + beta + theta; feature_vector.append(alpha / (den + EPSILON))
            elif feature_name == 'AB_Ratio': feature_vector.append(alpha / (beta + EPSILON))
            elif feature_name == 'Spec_Entropy': feature_vector.append(spec_entropy)
            
            # 這裡使用的就是上方濾波後算出的正確數值
            elif feature_name == 'Hjorth_Mobility': feature_vector.append(hjorth_mobility)
            elif feature_name == 'Hjorth_Complexity': feature_vector.append(hjorth_complexity)
            elif feature_name == 'Hjorth_Activity': feature_vector.append(var_0)
            
            else: feature_vector.append(0.0)
            
        all_features.append(feature_vector)
        
    return np.array(all_features)

# === Sequences ===
def create_sequences_from_groups(X_seg, y_seg, subjects_array, timesteps):
    # 注意：為了更嚴謹的 Block ID 控制，建議使用我上一次提供的 create_sequences_safe
    # 但如果我們已經在 segment_data_safely 中濾除了斷層，
    # 這裡使用 subjects_array 來切分序列也是 "相對" 安全的，
    # 唯一的風險是 Block 1 的結尾可能會連到 Block 2 的開頭。
    # 為了最簡單的修復，我們先沿用此邏輯，因為斷層間隔 (20s) 遠大於序列長度，影響較小。
    
    X_seq_list = []
    y_seq_list = []
    unique_subjects = np.unique(subjects_array)

    for subject in unique_subjects:
        mask = (subjects_array == subject)
        X_sub = X_seg[mask]
        y_sub = y_seg[mask]

        if len(X_sub) < timesteps: continue

        for i in range(len(X_sub) - timesteps + 1):
            X_seq_list.append(X_sub[i : i + timesteps])
            y_seq_list.append(y_sub[i + timesteps - 1])

    if not X_seq_list: return np.array([]), np.array([])
    return np.array(X_seq_list), np.array(y_seq_list)

# === Classifier ===
class EnhancedBCIClassifier:
    def __init__(self):
        self.model = None; self.history = None; self.scaler = StandardScaler()
        self.smooth_k = 5; self.vote_k = 7; self.threshold_ = 0.5

    def build_model(self, n_features, n_timesteps):
        model = Sequential(name="BCI_CNN_LSTM")
        model.add(Input(shape=(n_timesteps, n_features)))
        
        # CNN
        model.add(Conv1D(Config.CNN_FILTERS, Config.CNN_KERNEL_SIZE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        
        # Pooling layer
        # model.add(MaxPooling1D(pool_size=Config.CNN_POOL_SIZE)) # 池化可選
        
        # LSTM (CuDNN Enabled: recurrent_dropout=0)
        model.add(LSTM(Config.LSTM1_UNITS, recurrent_dropout=Config.LSTM1_RECURRENT_DROPOUT, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.5)) # 補償 Dropout

        model.add(LSTM(
            Config.LSTM2_UNITS, 
            recurrent_dropout=Config.LSTM2_RECURRENT_DROPOUT, 
            return_sequences=False, 
            name="LSTM_2"
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(Config.DENSE_UNITS, activation='relu'))
        model.add(Dropout(Config.DROPOUT))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(Config.LEARNING_RATE), metrics=['accuracy'])
        return model

    def fit(self, X_seq, y_seq):
        n, t, f = X_seq.shape
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, f)).reshape(n, t, f)
        
        # === Shuffle Training Data ===
        idx = np.arange(len(X_scaled))
        np.random.shuffle(idx) # 強制打亂訓練資料
        
        tf.keras.backend.clear_session()
        self.model = self.build_model(f, t)
        self.model.fit(
            X_scaled[idx], y_seq[idx],
            epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_FRACTION,
            callbacks=[EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOP_PATIENCE, restore_best_weights=True)],
            verbose=0
        )
        try:
            from sklearn.model_selection import StratifiedShuffleSplit  
            from sklearn.metrics import roc_curve
            
            # 從訓練資料中切出一小塊 (14%) 來找最佳閾值
            # 注意：這不是拿來訓練的，是拿來校準 "尺" 的
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.14, random_state=Config.RANDOM_STATE)
            (train_idx, val_idx) = next(sss.split(X_scaled, y_seq))
            
            # 預測機率
            val_proba = self.model.predict(X_scaled[val_idx], verbose=0).ravel()
            y_val = y_seq[val_idx]
            
            # 計算 ROC 曲線，找出最佳 J 值
            fpr, tpr, thr = roc_curve(y_val, val_proba)
            youden = tpr - fpr
            best = np.argmax(youden)
            
            # 設定最佳閾值
            if np.isfinite(thr[best]): 
                self.threshold_ = float(thr[best])
                print(f"Best Threshold found: {self.threshold_:.3f}") # 可選：印出來看看
            else: 
                self.threshold_ = 0.5
                
        except Exception as e:
            print(f"Warning: Failed to find threshold: {e}")
            self.threshold_ = 0.5
        return self

    def predict(self, X_seq):
        n, t, f = X_seq.shape
        X_scaled = self.scaler.transform(X_seq.reshape(-1, f)).reshape(n, t, f)
        prob = self.model.predict(X_scaled, verbose=0).ravel()
        
        # Post-process
        prob = np.convolve(prob, np.ones(self.smooth_k)/self.smooth_k, mode='same')
        pred = (prob > 0.5).astype(int)
        
        pad = self.vote_k // 2
        y_pad = np.pad(pred, (pad, pad), mode='edge')
        votes = np.convolve(y_pad, np.ones(self.vote_k), mode='valid')
        return (votes > (self.vote_k // 2)).astype(int)
        
    def get_loss(self):
        return self.model.history.history.get('loss', [])

# === Main Flow ===
def load_all_safe_data():
    all_f = []
    all_l = []
    all_s = []
    all_g = []
    folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    
    print(f"Loading and Segmenting data based on ranges: {Config.KEEP_RANGES}...")
    
    if not folders:
        print(f"Error: No subject folders found in {Config.DATASET_PATH}")
        return None, None, None

    for folder in folders:
        sid = os.path.basename(folder)
        d1, d2 = load_eeg_data(folder)
        if d1 is None or d2 is None:
            continue
        
        # === Use segment_data_safely ===
        s1, g1 = segment_data_safely(d1, Config.SAMPLING_RATE, Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO, Config.KEEP_RANGES, sid)
        s2, g2 = segment_data_safely(d2, Config.SAMPLING_RATE, Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO, Config.KEEP_RANGES, sid)
        # ===============================
        
        if len(s1)==0: continue
        
        f1 = extract_features(s1, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        f2 = extract_features(s2, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        
        all_f.append(np.vstack([f1, f2]))
        all_l.append(np.hstack([np.zeros(len(f1)), np.ones(len(f2))]))
        
        all_s.extend([sid]*len(np.hstack([np.zeros(len(f1)), np.ones(len(f2))])))
        
        g1_full = [f"{g}_R" for g in g1]
        g2_full = [f"{g}_F" for g in g2]
        all_g.extend(g1_full + g2_full)
    
    if not all_f: return None, None, None, None
    return np.vstack(all_f), np.hstack(all_l), np.array(all_s), np.array(all_g)

def run_safe_loso():
    print("Starting Leave-One-Subject-Out Cross-Validation")
    X_seg, y_seg, subjects, groups = load_all_safe_data()
    if X_seg is None:
        return None
    
    unique_subjects = sorted(list(set(subjects)))
    # random.shuffle(unique_subjects) # Optional shuffle
    
    # Results containers
    results = {
        'accuracies' : [],
        'confusion_matrices' : [],
        'loss_curves' : [],
        'subject_names' : []
    }
    
    for test_sub in unique_subjects:
        train_mask = (subjects != test_sub)
        test_mask = (subjects == test_sub)

        X_train, y_train = create_sequences_from_groups(X_seg[train_mask], y_seg[train_mask], groups[train_mask], Config.SEQUENCE_TIMESTEPS)
        X_test, y_test = create_sequences_from_groups(X_seg[test_mask], y_seg[test_mask], groups[test_mask], Config.SEQUENCE_TIMESTEPS)
        
        if len(X_train)==0: continue
        
        clf = EnhancedBCIClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        results['accuracies'].append(acc)
        results['confusion_matrices'].append(cm)
        results['loss_curves'].append(clf.get_loss())
        results['subject_names'].append(test_sub)
        
        print(f"{test_sub}: Accuracy = {acc:.3f}")
        
    return results

def plot_results(results):
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'BCI Classifier (Safe CNN-LSTM) - LOSO Results', fontsize=16)
    
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
    
    # 3. Loss Curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f'Fold {i+1}')
        axes[2].set_title('Training Loss Curves (First 5 Folds)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.savefig('bci_safe_results_full.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("BCI EEG Classification (Safe CNN-LSTM) - Relaxation vs Concentration"); print("=" * 70)
    tf.random.set_seed(Config.RANDOM_STATE)
    np.random.seed(Config.RANDOM_STATE)
    
    results = run_safe_loso()
    if results is None:
        print("Validation failed!")
        return
    
    mean_acc = np.mean(results['accuracies'])
    std_acc = np.std(results['accuracies'])
    print(f"\nOverall Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_acc = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        focus_acc = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        relax_prec = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        focus_prec = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0
        relax_f1 = 2 * (relax_prec * relax_acc) / (relax_prec + relax_acc) if (relax_prec + relax_acc) > 0 else 0
        focus_f1 = 2 * (focus_prec * focus_acc) / (focus_prec + focus_acc) if (focus_prec + focus_acc) > 0 else 0

    print(f"\nRelax Class:")
    print(f"  - Accuracy (Recall): {relax_acc:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_prec:.3f}")
    print(f"  - F1 Score: {relax_f1:.3f}")

    print(f"\nConcentration Class:")
    print(f"  - Accuracy (Recall): {focus_acc:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {focus_prec:.3f}")
    print(f"  - F1 Score: {focus_f1:.3f}")

    plot_results(results)
    print(f"\nResults saved to 'bci_safe_results_full.png'")

if __name__ == "__main__":
    main()
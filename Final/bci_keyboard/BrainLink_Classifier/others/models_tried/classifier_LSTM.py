"""
Brain-Computer Interface LSTM Classifier
For EEG signal relaxation/focus state classification
USE SEQUENCES OF DYNAMIC BANDPOWER RATIOS AS INPUT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import glob
import warnings
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy
from typing import List, Dict, Tuple

# === Keras / TensorFlow 匯入 ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    # === 修改：匯入 LSTM ===
    LSTM,  # <--- 使用 LSTM
    Dense, 
    Dropout, 
    BatchNormalization, 
    Input
    # (不再需要 Conv1D 和 GlobalAveragePooling1D)
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_113-2"

    # === 頻帶功率特徵設定 ===
    BANDS = {
        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [13, 30],
        'Gamma': [30, 45]
    }
    
    # 
    # --- 在此處啟用/停用您想要的特徵 ---
    #
    FEATURE_LIST = [
        'Alpha',        # 基礎相對功率
        'Beta',         # 基礎相對功率
        'AB_Ratio',     # Alpha / Beta
        'A_sum_ABT',    # α / (α + β + θ)
        'B_sum_AT',     # β / (α + θ)
        'A_sum_BT',      # α / (β + θ)

        'Spec_Entropy',      # 頻譜熵
        'Hjorth_Mobility',   # Hjorth 移動性
        'Hjorth_Complexity'  # Hjorth 複雜度
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST) # 自動計算
    
    # === Signal processing ===
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 2.5               
    OVERLAP_RATIO = 0.9                
    
    # === 修改：模型與序列設定 ===
    # 
    # --- 序列 (Sequence) 超參數 ---
    #
    # SEQUENCE_TIMESTEPS:
    #   最重要的參數。定義模型 "一次" 看多少個連續的窗口 (segments)。
    #   例如：TIMESTEPS = 15, 步長 0.25s -> 總 "記憶" 跨度約 6s
    #
    SEQUENCE_TIMESTEPS = 15      # !(可調) 
    
    #
    # --- LSTM 超參數 (Hyperparameters) ---
    #
    # LSTM_UNITS:
    #   LSTM 層的神經元數量 (記憶容量)。(例如 32, 64, 128)
    #   數量太少可能學不到複雜模式，太多則容易過擬合。
    #
    # LSTM_RECURRENT_DROPOUT:
    #   在 LSTM 內部時間步之間應用的 Dropout。
    #   這是防止 LSTM 過擬合的 *關鍵* 參數。(例如 0.1, 0.2, 0.3)
    #
    # DENSE_UNITS:
    #   在 LSTM 輸出之後，全連接層的神經元數量。(例如 32, 64)
    #
    # DROPOUT:
    #   在全連接層之後應用的 Dropout。(例如 0.3, 0.5)
    #
    # --- 推薦參數組合 ---
    # 
    # --- 組合 1: 輕量 (推薦) ---
    # LSTM_UNITS = 64
    # LSTM_RECURRENT_DROPOUT = 0.2
    # DENSE_UNITS = 32
    # DROPOUT = 0.5
    #
    # --- 組合 2: 較深 (可選) ---
    # (修改 build_model 加入第二層 LSTM(64, ...))
    # LSTM_UNITS = 64
    # DENSE_UNITS = 64
    # DROPOUT = 0.5
    #
    # --- 組合 3: 較大容量 ---
    LSTM_UNITS = 128
    LSTM_RECURRENT_DROPOUT = 0.3
    DENSE_UNITS = 64
    DROPOUT = 0.5
    # ----------------------------------------------------
    
    # --- 其他超參數 ---
    LEARNING_RATE = 0.0005       # !(可調) 學習率
    EPOCHS = 100                 # !(可調) 最大週期
    BATCH_SIZE = 16              # !(可調) 批次大小
    EARLY_STOP_PATIENCE = 15     # !(可調) 早停耐心值 (可設 15 或 20)
    VALIDATION_FRACTION = 0.1    # !(可調) 驗證集切分比例
    
    # Feature selection (不建議與 Keras 混用)
    FEATURE_SELECTION = False          
    N_FEATURES_SELECT = N_FEATURES_TOTAL 
    
    # Other settings
    RANDOM_STATE = 42

# === student preprocessing ===
def preprocess_signal(data):
    fs = 500; seg_len = int(round(20.0 * fs)); gap_len = int(round(20.0 * fs))
    start = int(round(20.0 * fs)); max_end = min(len(data), int(round(400.0 * fs)))
    valid_idx = [];
    while start + seg_len <= max_end:
        valid_idx.append((start, start + seg_len)); start += seg_len + gap_len
    if not valid_idx: x = np.asarray(data, dtype=float)
    else: x = np.concatenate([data[s:e] for s, e in valid_idx], axis=0).astype(float)
    x = x - np.median(x)
    dc_window_sec = 2.0; w = int(round(dc_window_sec * fs))
    if w >= 2 and len(x) > w:
        kernel = np.ones(w, dtype=float) / float(w)
        baseline = np.convolve(x, kernel, mode='same'); x = x - baseline
    return x

# === load_eeg_data ===
def load_eeg_data(subject_path):
    relax_file = os.path.join(subject_path, "1.txt"); focus_file = os.path.join(subject_path, "2.txt")
    try:
        relax_data = np.loadtxt(relax_file); focus_data = np.loadtxt(focus_file)
        relax_data = preprocess_signal(relax_data); focus_data = preprocess_signal(focus_data)
        return relax_data, focus_data
    except Exception as e:
        print(f"Error loading data for {subject_path}: {e}"); return None, None

# === create_segments ===
def create_segments(data, segment_length_samples, overlap_samples):
    if len(data) < segment_length_samples: return np.array([])
    segments = []; start = 0; step = segment_length_samples - overlap_samples
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]; segments.append(segment); start += step
    return np.array(segments)

# === 頻帶功率基礎函數 ===
def get_band_power(epoch_data, fs, bands):
    freqs, psd = signal.welch(epoch_data, fs, nperseg=len(epoch_data), window='hann')
    freq_res = freqs[1] - freqs[0]; abs_powers = {}
    for band_name, (low_hz, high_hz) in bands.items():
        idx_band = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
        if len(idx_band) == 0: abs_powers[band_name] = 0; continue
        power = simpson(psd[idx_band], dx=freq_res)
        abs_powers[band_name] = power
    total_power = sum(abs_powers.values())
    if total_power == 0: return {band: 0 for band in bands}
    rel_powers = {band: power / total_power for band, power in abs_powers.items()}
    return rel_powers, psd

# === 動態特徵提取函數 ===
def extract_features(segments, fs, bands, feature_list):
    band_names_ordered = list(bands.keys()); all_features = []; EPSILON = 1e-10 
    
    for segment in segments:
        
        # --- 1. 計算頻帶特徵 ---
        rel_powers, psd = get_band_power(segment, fs, bands)

        # --- 2. 計算新特徵 (時域/頻譜複雜度) ---
        
        # 2a. 頻譜熵 (Spectral Entropy)
        # 先將 PSD 歸一化，使其像一個機率分佈
        psd_norm = psd / (np.sum(psd) + EPSILON) 
        spec_entropy = entropy(psd_norm)

        # 2b. Hjorth 參數 (直接在原始 segment 上計算)
        d1 = np.diff(segment) # 一階差分
        d2 = np.diff(d1)      # 二階差分
        
        var_0 = np.var(segment) # Activity (原始訊號變異數)
        var_d1 = np.var(d1)     # Mobility (一階差分變異數)
        var_d2 = np.var(d2)     # Complexity (二階差分變異數)
        
        hjorth_mobility = np.sqrt(var_d1 / (var_0 + EPSILON))
        
        # 複雜度的另一種計算方式 (Mobility of the 1st derivative)
        mobility_d1 = np.sqrt(var_d2 / (var_d1 + EPSILON)) 
        # Complexity (Mobility of d1 / Mobility of 0)
        hjorth_complexity = mobility_d1 / (hjorth_mobility + EPSILON)
        
        # --- 3. 獲取基礎頻帶值 ---
        delta = rel_powers.get('Delta', 0); theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0); beta  = rel_powers.get('Beta', 0)
        gamma = rel_powers.get('Gamma', 0); feature_vector = []
        
        # --- 4. 根據 Feature_List 建立特徵向量 ---
        # ! (可調) 把不要的特徵加上註解
        for feature_name in feature_list:
            if feature_name == 'Delta': feature_vector.append(delta)
            elif feature_name == 'Theta': feature_vector.append(theta)
            elif feature_name == 'Alpha': feature_vector.append(alpha)
            elif feature_name == 'Beta': feature_vector.append(beta)
            # elif feature_name == 'Gamma': feature_vector.append(gamma)
            elif feature_name == 'A_sum_ABT': den = alpha + beta + theta; feature_vector.append(alpha / (den + EPSILON))
            elif feature_name == 'B_sum_AT': den = alpha + theta; feature_vector.append(beta / (den + EPSILON))
            elif feature_name == 'A_sum_BT': den = beta + theta; feature_vector.append(alpha / (den + EPSILON))
            elif feature_name == 'AB_Ratio': feature_vector.append(alpha / (beta + EPSILON))
            elif feature_name == 'Spec_Entropy': feature_vector.append(spec_entropy)
            elif feature_name == 'Hjorth_Mobility': feature_vector.append(hjorth_mobility)
            elif feature_name == 'Hjorth_Complexity': feature_vector.append(hjorth_complexity)
            else: warnings.warn(f"Unknown feature: {feature_name}"); feature_vector.append(0.0)
                
        all_features.append(feature_vector)
        
    return np.array(all_features)
# ==================================

# === Keras 數據準備函數 ===
def create_sequences_from_groups(X_seg, y_seg, subjects_array, timesteps):
    X_seq_list = []; y_seq_list = []
    unique_subjects = np.unique(subjects_array)
    
    for subject in unique_subjects:
        subject_mask = (subjects_array == subject)
        X_subject = X_seg[subject_mask]; y_subject = y_seg[subject_mask]
        
        if len(X_subject) < timesteps:
            continue
            
        for i in range(len(X_subject) - timesteps + 1):
            X_seq_list.append(X_subject[i : i + timesteps])
            y_seq_list.append(y_subject[i + timesteps - 1])
            
    if not X_seq_list:
        return np.array([]), np.array([])
        
    return np.array(X_seq_list), np.array(y_seq_list)

# === 重寫：Keras 版本的 BCI 分類器 ===
class EnhancedBCIClassifier:
    """Enhanced Brain-Computer Interface Classifier (LSTM Keras Wrapper)"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.feature_selector = None 
        self.threshold_ = 0.5
        self.smooth_k = 5
        self.vote_k = 7

    def build_model(self, n_features, n_timesteps):
        """建立 LSTM Keras 模型 (單層架構)"""
        
        # 從 Config 讀取超參數
        lstm_units = Config.LSTM_UNITS
        recurrent_dropout = Config.LSTM_RECURRENT_DROPOUT
        dense_units = Config.DENSE_UNITS
        dropout_rate = Config.DROPOUT
        learning_rate = Config.LEARNING_RATE
        
        model = Sequential(name="BCI_LSTM")
        model.add(Input(shape=(n_timesteps, n_features), name="Input_Sequence"))
        
        # !--- (可選) BatchNormalization on Input ---
        model.add(BatchNormalization())

        # --- LSTM 層 ---
        # 這是模型的核心。它會 "讀取" 整個序列
        # return_sequences=False (預設) 是關鍵，
        # 因為它只會輸出 *最後一個時間步* 的隱藏狀態，
        # 這就是我們用來分類的 "記憶總結"。
        
        # !--- (可選) 單層 LSTM ---
        # --- 最後一層 LSTM 將 return_sequences 設為 False
        # model.add(LSTM(units=lstm_units, recurrent_dropout=recurrent_dropout, return_sequences=False, name="LSTM_1"))
        
        # !--- (可選) 堆疊 LSTM ---
        # 如果要堆疊，第一層必須 return_sequences=True
        model.add(LSTM(units=lstm_units, recurrent_dropout=recurrent_dropout, return_sequences=True, name="LSTM_1"))
        model.add(LSTM(units=lstm_units, recurrent_dropout=recurrent_dropout, return_sequences=False, name="LSTM_2"))
        # --------------------------
        
        # BatchNormalization 幫助穩定訓練
        model.add(BatchNormalization(name="BatchNorm_1"))
        
        # --- 全連接層 (MLP) ---
        # ! (可選) Activation function: ReLU, leaky_ReLU or GeLU
        model.add(Dense(dense_units, activation='leaky_relu', name="Dense_1"))
        model.add(Dropout(dropout_rate, name="Dropout"))
        
        # --- S 輸出層 ---
        model.add(Dense(1, activation='sigmoid', name="Output_Sigmoid"))
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model

    def fit(self, X_seq, y_seq):
        """訓練 Keras 模型 (與 CNN 版本相同)"""
        n_samples, n_timesteps, n_features = X_seq.shape
        
        # === Scaling ===
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        X_selected = X_scaled_seq

        tf.keras.backend.clear_session()
        self.model = self.build_model(n_features, n_timesteps)
        
        # === Early Stopping ===
        early_stopping_cb = EarlyStopping(
            monitor='val_loss', 
            patience=Config.EARLY_STOP_PATIENCE,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_selected,
            y_seq,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_FRACTION, 
            callbacks=[early_stopping_cb],
            verbose=0
        )
        
        # === Threshold tuning ===
        try:
            from sklearn.model_selection import StratifiedShuffleSplit  
            from sklearn.metrics import roc_curve
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.14, random_state=Config.RANDOM_STATE)
            (train_idx, val_idx) = next(sss.split(X_selected, y_seq))
            val_proba = self.model.predict(X_selected[val_idx]).ravel()
            y_val = y_seq[val_idx]
            
            fpr, tpr, thr = roc_curve(y_val, val_proba)
            youden = tpr - fpr; best = np.argmax(youden)
            if np.isfinite(thr[best]): self.threshold_ = float(thr[best])
            else: self.threshold_ = 0.5
        except Exception as e:
            print(f"Warning: Failed to find threshold: {e}")
            self.threshold_ = 0.5
        return self

    # === Voting ===
    def _majority_vote(self, y_pred: np.ndarray, k: int) -> np.ndarray:
        if k <= 1: return y_pred
        if k % 2 == 0: k += 1
        pad = k // 2; y_pad = np.pad(y_pred.astype(int), (pad, pad), mode='edge')
        votes = np.convolve(y_pad, np.ones(k, dtype=int), mode='valid')
        return (votes > (k // 2)).astype(int)

    def _postprocess_predict(self, proba_seq: np.ndarray) -> np.ndarray:
        proba = proba_seq
        k_s = int(getattr(self, "smooth_k", 1) or 1)
        if k_s > 1:
            kernel = np.ones(k_s, dtype=float) / float(k_s)
            proba = np.convolve(proba, kernel, mode='same')
        thr = self.threshold_ if np.isfinite(getattr(self, "threshold_", np.nan)) else 0.5
        y_pred = (proba > thr).astype(int)
        k_v = int(getattr(self, "vote_k", 1) or 1)
        y_pred = self._majority_vote(y_pred, k_v)
        return y_pred

    def predict(self, X_seq):
        """使用 Keras 模型和後處理進行預測 (與 CNN 版本相同)"""
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        X_selected = X_scaled_seq
        raw_probabilities = self.model.predict(X_selected).ravel()
        return self._postprocess_predict(raw_probabilities)       

    def get_loss_curve(self):
        """Get the loss curve (from Keras history) (與 CNN 版本相同)"""
        if self.history and 'loss' in self.history.history:
            return self.history.history['loss']
        return []
    # ============================

# === 主執行函數 ===

# 步驟 1: 載入所有 "segments"
def load_all_segments_and_features():
    all_features = []; all_labels = []; all_subjects = []
    subject_folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    if not subject_folders:
        print(f"Error: No subject folders found in {Config.DATASET_PATH}"); return None, None, None
    
    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_data, focus_data = load_eeg_data(subject_folder)
        if relax_data is None or focus_data is None: continue

        segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
        overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)
        relax_segments = create_segments(relax_data, segment_length_samples, overlap_samples)
        focus_segments = create_segments(focus_data, segment_length_samples, overlap_samples)
        
        if relax_segments.size == 0 or focus_segments.size == 0:
            print(f"Warning: Not enough data for {subject_id}. Skipping..."); continue
            
        relax_features = extract_features(relax_segments, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        focus_features = extract_features(focus_segments, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        
        relax_labels = np.zeros(len(relax_features)); focus_labels = np.ones(len(focus_features))
        subject_features = np.vstack([relax_features, focus_features])
        subject_labels = np.hstack([relax_labels, focus_labels])
        subject_ids = [subject_id] * len(subject_labels)
        
        all_features.append(subject_features); all_labels.append(subject_labels); all_subjects.extend(subject_ids)
    
    if not all_features: print("Error: No valid data found"); return None, None, None
    
    X_seg = np.vstack(all_features); y_seg = np.hstack(all_labels); subjects_array = np.array(all_subjects)
    return X_seg, y_seg, subjects_array

# 步驟 2: LOSO 驗證
def leave_one_subject_out_validation():
    # --- 修改標題 ---
    print(f"Starting LOSO Cross-Validation (LSTM) with {Config.N_FEATURES_TOTAL} features...")
    print(f"Features: {Config.FEATURE_LIST}")
    print(f"Sequence: {Config.SEQUENCE_TIMESTEPS} timesteps, Segment: {Config.SEGMENT_LENGTH}s, Overlap: {Config.OVERLAP_RATIO}")
    # --------------
    
    X_seg, y_seg, subjects_array = load_all_segments_and_features()
    if X_seg is None: return None
    
    unique_subjects = sorted(list(set(np.unique(subjects_array))))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    for test_subject in unique_subjects:
        train_mask = (subjects_array != test_subject)
        test_mask = (subjects_array == test_subject)
        X_train_seg, y_train_seg, subjects_train = X_seg[train_mask], y_seg[train_mask], subjects_array[train_mask]
        X_test_seg, y_test_seg, subjects_test = X_seg[test_mask], y_seg[test_mask], subjects_array[test_mask]
        
        # --- 使用 SEQUENCE_TIMESTEPS ---
        X_train, y_train = create_sequences_from_groups(X_train_seg, y_train_seg, subjects_train, Config.SEQUENCE_TIMESTEPS)
        X_test, y_test = create_sequences_from_groups(X_test_seg, y_test_seg, subjects_test, Config.SEQUENCE_TIMESTEPS)
        # -----------------------------------
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping {test_subject}: Not enough data after creating sequences.")
            continue
            
        classifier = EnhancedBCIClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        results['accuracies'].append(accuracy); results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve()); results['subject_names'].append(test_subject)
        
        print(f"{test_subject}: Accuracy = {accuracy:.3f} (N_train={len(X_train)}, N_test={len(X_test)})")
    
    return results

# 步驟 3: 繪圖
def plot_results(results):
    if results is None: return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # --- 修改標題 ---
    fig.suptitle(f'BCI Classifier (LSTM - {Config.N_FEATURES_TOTAL} Features) - LOSO Results', fontsize=16)
    # --------------
    
    # 1. Accuracy
    axes[0].bar(range(len(results['accuracies'])), results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.6 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_xlabel('Subject Index'); axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1)
    
    # 2. Confusion Matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus'], yticklabels=['Relax', 'Focus'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    
    # 3. Training loss curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f'Fold {i+1}')
        axes[2].set_title('Training Loss Curves (First 5 Folds)')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss (binary_crossentropy)')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves available', 
                     horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Training Loss Curves')
    
    plt.tight_layout()
    plt.savefig('bci_results_lstm_bandpower.png', dpi=300, bbox_inches='tight')
    plt.show()

# 步驟 4: Main 函數
def main():
    print("BCI EEG Classification (LSTM - Sequential Features) - Relaxation vs Concentration")
    print("=" * 70)
    
    tf.random.set_seed(Config.RANDOM_STATE)
    np.random.seed(Config.RANDOM_STATE)
    
    results = leave_one_subject_out_validation()
    if results is None: print("Validation failed!"); return
    
    mean_accuracy = np.mean(results['accuracies']); std_accuracy = np.std(results['accuracies'])
    print(f"\nOverall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        concentration_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        concentration_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0

    print(f"\nRelax Class:"); print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})"); print(f"  - Precision: {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")
    print(f"\nConcentration Class:"); print(f"  - Accuracy (Recall): {concentration_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})"); print(f"  - Precision: {concentration_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")
    plot_results(results)
    print(f"\nResults saved to 'bci_results_lstm_bandpower.png'")

if __name__ == "__main__":
    main()
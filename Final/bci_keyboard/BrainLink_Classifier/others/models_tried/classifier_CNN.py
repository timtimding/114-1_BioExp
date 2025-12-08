"""
Brain-Computer Interface 1D-CNN Classifier
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
from typing import List, Dict, Tuple

# === 新增：Keras / TensorFlow 匯入 ===
# 確保你已經 `pip install tensorflow`
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, 
    Dense, 
    Dropout, 
    BatchNormalization, 
    GlobalAveragePooling1D,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# ====================================

# (我們不再需要 sklearn 的模型，但保留匯入以便切換)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

warnings.filterwarnings('ignore')

class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_113-2"

    # === 頻帶功率特徵設定 ===
    BANDS = {
        'Delta': [1, 5],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [12, 30],
        'Gamma': [30, 45]
    }
    
    # 動態特徵列表 (在此處定義特徵)
    FEATURE_LIST = [
        'AB_Ratio',     # Alpha / Beta
        'A_sum_ABT',    # α / (α + β + θ)
        'B_sum_AT',     # β / (α + θ)
        'A_sum_BT'      # α / (β + θ)
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST)
    
    # === Signal processing (您修改後的即時設定) ===
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 6
    OVERLAP_RATIO = 0.8                
    
    # === 關鍵：CNN 模型設定 ===
    # ------------------------------------------------------------------
    # 參數說明 (Hyperparameter)
    # ------------------------------------------------------------------
    #
    # CNN_TIMESTEPS:
    #   這是最重要的參數。它定義了 CNN "一次" 看多少個連續的 2.5s 窗口。
    #   例如：TIMESTEPS = 10， OVERLAP_RATIO = 0.9 (步長 0.25s)
    #   這意味著 CNN 的 "記憶" 跨度是 (10-1)*0.25s + 2.5s = 4.75s 
    #   (它看了 10 個窗口，總共跨了 4.75s 的時間)
    #
    # CNN_FILTERS: 
    #   卷積核的數量。偵測 "特徵" 的數量。 (例如 16, 32, 64)
    #
    # CNN_KERNEL_SIZE:
    #   卷積核的大小 (在時間維度上)。例如 3，代表它一次看 3 個時間步 (timesteps)。
    #
    # CNN_DENSE_UNITS:
    #   在卷積層之後，全連接層的神經元數量。(例如 32, 64)
    #
    # CNN_DROPOUT:
    #   防止過擬合的丟棄率 (例如 0.3 = 丟棄 30%)
    #
    # CNN_EPOCHS:
    #   最大訓練週期。我們使用 EarlyStopping，所以設高一點沒關係。
    #
    # ------------------------------------------------------------------
    # 推薦參數組合
    # ------------------------------------------------------------------
    #
    # --- 組合 1: 快速、輕量 (適合初步測試) ---
    # CNN_TIMESTEPS = 10           # 總 "記憶" 跨度約 4.75s
    # CNN_FILTERS = 16
    # CNN_KERNEL_SIZE = 3
    # CNN_DENSE_UNITS = 32
    # CNN_DROPOUT = 0.3
    #
    # --- 組合 2: 中等、較深 (推薦) ---
    # CNN_TIMESTEPS = 15           # 總 "記憶" 跨度約 6s
    # CNN_FILTERS = 32
    # CNN_KERNEL_SIZE = 3
    # CNN_DENSE_UNITS = 64
    # CNN_DROPOUT = 0.5
    #
    # --- 組合 3: 較長記憶、更深 (可選) ---
    # CNN_TIMESTEPS = 20           # 總 "記憶" 跨度約 7.25s
    # CNN_FILTERS = 64
    # CNN_KERNEL_SIZE = 5          # 搭配較長的記憶，使用較大的 kernel
    # CNN_DENSE_UNITS = 64
    # CNN_DROPOUT = 0.5
    #
    # --- 組合 4: 雙層 CNN (可選，修改 build_model 函數) ---
    # CNN_TIMESTEPS = 15
    # (修改 build_model 加入第二層 Conv1D(64, 3))
    #
    # --- 組合 5: 我們根據實作狀況調整超參數 ---
    CNN_TIMESTEPS = 15           # 總 "記憶" 跨度約 6s
    CNN_FILTERS_L1 = 32
    CNN_FILTERS_L2 = 64
    CNN_KERNEL_SIZE = 5
    CNN_DENSE_UNITS = 64
    CNN_DROPOUT = 0.5

    # --- 其他超參數 ---
    CNN_LEARNING_RATE = 0.0005      # 控制模型在每次更新時權重變化的幅度
    CNN_EPOCHS = 100                # 最大週期
    CNN_BATCH_SIZE = 64             # 一次「看」多少個訓練樣本
    CNN_EARLY_STOP_PATIENCE = 10    # 10 個 epoch 沒改善就停止
    
    # Feature selection (CNN/Keras 搭配 Sklearn 的 SelectKBest 很麻煩，建議關閉)
    FEATURE_SELECTION = False          
    N_FEATURES_SELECT = N_FEATURES_TOTAL 
    
    # Other settings
    RANDOM_STATE = 42
    VALIDATION_FRACTION = 0.1   # 用於 Keras 的 .fit() 驗證集切分


# === student preprocessing (保持不變) ===
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

# === load_eeg_data (保持不變) ===
def load_eeg_data(subject_path):
    relax_file = os.path.join(subject_path, "1.txt"); focus_file = os.path.join(subject_path, "2.txt")
    try:
        relax_data = np.loadtxt(relax_file); focus_data = np.loadtxt(focus_file)
        relax_data = preprocess_signal(relax_data); focus_data = preprocess_signal(focus_data)
        return relax_data, focus_data
    except Exception as e:
        print(f"Error loading data for {subject_path}: {e}"); return None, None

# === create_segments (保持不變) ===
def create_segments(data, segment_length_samples, overlap_samples):
    if len(data) < segment_length_samples: return np.array([])
    segments = []; start = 0; step = segment_length_samples - overlap_samples
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]; segments.append(segment); start += step
    return np.array(segments)

# === 頻帶功率基礎函數 (保持不變) ===
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
    return rel_powers

# === 動態特徵提取函數 (保持不變) ===
def extract_features(segments, fs, bands, feature_list):
    band_names_ordered = list(bands.keys()); all_features = []; EPSILON = 1e-10 
    for segment in segments:
        rel_powers = get_band_power(segment, fs, bands)
        delta = rel_powers.get('Delta', 0); theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0); beta  = rel_powers.get('Beta', 0)
        gamma = rel_powers.get('Gamma', 0); feature_vector = []
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
            else: warnings.warn(f"Unknown feature: {feature_name}"); feature_vector.append(0.0)
        all_features.append(feature_vector)
    return np.array(all_features)
# ==================================

# === 新增：Keras 數據準備函數 ===
def create_sequences_from_groups(X_seg, y_seg, subjects_array, timesteps):
    """
    將 (N_segments, N_features) 數據轉換為 (N_sequences, timesteps, N_features)。
    此函數會確保序列不會跨越不同的受試者。
    """
    X_seq_list = []
    y_seq_list = []
    
    # 找出所有獨立的受試者
    unique_subjects = np.unique(subjects_array)
    
    for subject in unique_subjects:
        # 提取這位受試者的所有 segments
        subject_mask = (subjects_array == subject)
        X_subject = X_seg[subject_mask]
        y_subject = y_seg[subject_mask]
        
        # 如果這位受試者的 segments 數量不足以創建一個序列，則跳過
        if len(X_subject) < timesteps:
            continue
            
        # 在這位受試者的數據內部創建滑動窗口序列
        for i in range(len(X_subject) - timesteps + 1):
            # 一個序列 (X)
            X_seq_list.append(X_subject[i : i + timesteps])
            # 序列的標籤 (y) - 我們使用序列中 *最後一個* 時間點的標籤
            y_seq_list.append(y_subject[i + timesteps - 1])
            
    if not X_seq_list:
        # 如果數據太少，返回空的陣列
        return np.array([]), np.array([])
        
    return np.array(X_seq_list), np.array(y_seq_list)

# === 重寫：Keras 版本的 BCI 分類器 ===
class EnhancedBCIClassifier:
    """Enhanced Brain-Computer Interface Classifier (1D-CNN Keras Wrapper)"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.feature_selector = None 
        self.threshold_ = 0.5
        self.smooth_k = 5
        self.vote_k = 7

    # === ↓↓↓ 關鍵修改：build_model 函數 ↓↓↓ ===
    def build_model(self, n_features, n_timesteps):
        """建立 1D-CNN Keras 模型 (雙層架構)"""
        
        # 從 Config 讀取超參數
        filters_l1 = Config.CNN_FILTERS_L1
        filters_l2 = Config.CNN_FILTERS_L2
        kernel_size = Config.CNN_KERNEL_SIZE
        dense_units = Config.CNN_DENSE_UNITS
        dropout_rate = Config.CNN_DROPOUT
        learning_rate = Config.CNN_LEARNING_RATE
        
        model = Sequential(name="BCI_1D_CNN_L2")
        model.add(Input(shape=(n_timesteps, n_features), name="Input_Sequence"))
        
        # --- 卷積層 1 ---
        model.add(Conv1D(
            filters=filters_l1, 
            kernel_size=kernel_size, 
            activation='relu', 
            padding='same', # 'same' 確保輸出序列長度不變
            name="Conv1D_1"
        ))
        model.add(BatchNormalization(name="BatchNorm_1"))
        
        # --- (新增) 卷積層 2 ---
        model.add(Conv1D(
            filters=filters_l2, 
            kernel_size=kernel_size, 
            activation='relu', 
            padding='same',
            name="Conv1D_2"
        ))
        model.add(BatchNormalization(name="BatchNorm_2"))
        # --- (新增結束) ---
        
        # --- 池化層 ---
        # GlobalAveragePooling1D 會計算整個序列的平均值
        model.add(GlobalAveragePooling1D(name="GlobalAvgPool"))
        
        # --- 全連接層 (MLP) ---
        model.add(Dense(dense_units, activation='relu', name="Dense_1"))
        model.add(Dropout(dropout_rate, name="Dropout"))
        
        # --- 輸出層 ---
        model.add(Dense(1, activation='sigmoid', name="Output_Sigmoid"))
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model

    def fit(self, X_seq, y_seq):
        """訓練 Keras 模型"""
        n_samples, n_timesteps, n_features = X_seq.shape
        
        # === 1. 標準化 (Scaling) ===
        # Keras 需要 3D 輸入，但 Scaler 需要 2D (N_samples * N_features)
        # 1. 攤平 -> fit_transform -> 2. 重塑回 3D
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        
        X_selected = X_scaled_seq # (我們在此禁用了特徵選擇)
        
        # === 2. 建立模型 ===
        tf.keras.backend.clear_session() # 清除舊模型
        self.model = self.build_model(n_features, n_timesteps)
        # self.model.summary() # (可選) 印出模型架構
        
        # === 3. 設定 Early Stopping ===
        # 監控 'val_loss' (驗證集損失)
        early_stopping_cb = EarlyStopping(
            monitor='val_loss', 
            patience=Config.CNN_EARLY_STOP_PATIENCE,
            restore_best_weights=True # 關鍵：結束時恢復到最佳權重
        )

        # === 4. 訓練模型 ===
        self.history = self.model.fit(
            X_selected,
            y_seq,
            epochs=Config.CNN_EPOCHS,
            batch_size=Config.CNN_BATCH_SIZE,
            validation_split=Config.VALIDATION_FRACTION, # Keras 會自動切分 10% (0.1) 作為驗證集
            callbacks=[early_stopping_cb],
            verbose=0 # (設為 1 可看訓練過程)
        )
        
        # === 5. student postprocessing (尋找最佳閾值) ===
        # (這部分邏輯與您先前版本相同，但現在應用於 Keras 模型)
        try:
            from sklearn.model_selection import StratifiedShuffleSplit  
            from sklearn.metrics import roc_curve
            
            # 建立一個小型的驗證集 (14%)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.14, random_state=Config.RANDOM_STATE)
            # 注意：我們必須從 X_selected (Keras 輸入) 和 y_seq 中分割
            (train_idx, val_idx) = next(sss.split(X_selected, y_seq))
            
            # 在這個驗證集上取得 Keras 模型的機率預測
            # .predict() 輸出 (N, 1), .ravel() 轉為 1D 陣列
            val_proba = self.model.predict(X_selected[val_idx]).ravel()
            y_val = y_seq[val_idx]
            
            # 尋找 Youden's J
            fpr, tpr, thr = roc_curve(y_val, val_proba)
            youden = tpr - fpr; best = np.argmax(youden)
            if np.isfinite(thr[best]): self.threshold_ = float(thr[best])
            else: self.threshold_ = 0.5
        except Exception as e:
            print(f"Warning: Failed to find threshold: {e}")
            self.threshold_ = 0.5
        # =======================================
        return self

    # === student postprocessing (平滑/投票) (保持不變) ===
    def _majority_vote(self, y_pred: np.ndarray, k: int) -> np.ndarray:
        if k <= 1: return y_pred
        if k % 2 == 0: k += 1
        pad = k // 2; y_pad = np.pad(y_pred.astype(int), (pad, pad), mode='edge')
        votes = np.convolve(y_pad, np.ones(k, dtype=int), mode='valid')
        return (votes > (k // 2)).astype(int)

    # === 修改：_postprocess_predict 以接收 Keras 原始機率 ===
    def _postprocess_predict(self, proba_seq: np.ndarray) -> np.ndarray:
        """
        Calibration → smoothing → thresholding → majority vote.
        接收 Keras 模型的原始機率 (1D 陣列)
        """
        proba = proba_seq # (Keras 輸出已是 1D 機率)

        # (平滑機率)
        k_s = int(getattr(self, "smooth_k", 1) or 1)
        if k_s > 1:
            kernel = np.ones(k_s, dtype=float) / float(k_s)
            proba = np.convolve(proba, kernel, mode='same')

        # (使用 self.threshold_ 進行閾值化)
        thr = self.threshold_ if np.isfinite(getattr(self, "threshold_", np.nan)) else 0.5
        y_pred = (proba > thr).astype(int)

        # (多數投票)
        k_v = int(getattr(self, "vote_k", 1) or 1)
        y_pred = self._majority_vote(y_pred, k_v)
        return y_pred
    # =======================================

    def predict(self, X_seq):
        """使用 Keras 模型和後處理進行預測"""
        
        # === 1. 標準化 (Scaling) ===
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        # ** 使用 .transform() 而非 .fit_transform() **
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        
        X_selected = X_scaled_seq # (禁用特徵選擇)
        
        # === 2. 獲取 Keras 原始機率 ===
        # .predict() 輸出 (N, 1), .ravel() 轉為 1D
        raw_probabilities = self.model.predict(X_selected).ravel()
        
        # === 3. 呼叫 student postprocessing === 
        return self._postprocess_predict(raw_probabilities)       
        # =======================================

    # === 修改：從 Keras history 獲取 loss ===
    def get_loss_curve(self):
        """Get the loss curve (from Keras history)"""
        if self.history and 'loss' in self.history.history:
            return self.history.history['loss']
        return [] # 如果沒有訓練，返回空列表
    # ============================

# === 修改：主執行函數 ===

# 步驟 1: 載入所有 "segments" (2.5s 窗口)
def load_all_segments_and_features():
    """
    載入所有受試者的數據，並將其轉換為 "segments" (特徵向量)。
    返回: X_seg, y_seg, subjects_array
    (N_total_segments, N_features), (N_total_segments,), (N_total_segments,)
    """
    all_features = []
    all_labels = []
    all_subjects = []
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

# 步驟 2: LOSO 驗證 (修改)
def leave_one_subject_out_validation():
    """Perform Leave-One-Subject-Out cross-validation (for 1D-CNN)"""
    print(f"Starting LOSO Cross-Validation (1D-CNN) with {Config.N_FEATURES_TOTAL} features...")
    print(f"Features: {Config.FEATURE_LIST}")
    print(f"Sequence: {Config.CNN_TIMESTEPS} timesteps, Segment: {Config.SEGMENT_LENGTH}s, Overlap: {Config.OVERLAP_RATIO}")
    
    # 1. 載入所有 "segments" (2D 數據)
    X_seg, y_seg, subjects_array = load_all_segments_and_features()
    if X_seg is None: return None
    
    unique_subjects = sorted(list(set(np.unique(subjects_array))))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    for test_subject in unique_subjects:
        # 2. 分割 2D 數據
        train_mask = (subjects_array != test_subject)
        test_mask = (subjects_array == test_subject)
        
        X_train_seg, y_train_seg, subjects_train = X_seg[train_mask], y_seg[train_mask], subjects_array[train_mask]
        X_test_seg, y_test_seg, subjects_test = X_seg[test_mask], y_seg[test_mask], subjects_array[test_mask]
        
        # 3. 將 2D segments 轉換為 3D Keras 序列
        # 這是關鍵步驟！
        X_train, y_train = create_sequences_from_groups(X_train_seg, y_train_seg, subjects_train, Config.CNN_TIMESTEPS)
        X_test, y_test = create_sequences_from_groups(X_test_seg, y_test_seg, subjects_test, Config.CNN_TIMESTEPS)
        
        # 檢查是否有足夠的數據
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping {test_subject}: Not enough data after creating sequences.")
            continue
            
        # 4. 訓練 Keras 模型
        classifier = EnhancedBCIClassifier()
        classifier.fit(X_train, y_train)
        
        # 5. 預測
        y_pred = classifier.predict(X_test)
        
        # 6. 儲存結果
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        results['accuracies'].append(accuracy); results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve()); results['subject_names'].append(test_subject)
        
        print(f"{test_subject}: Accuracy = {accuracy:.3f} (N_train={len(X_train)}, N_test={len(X_test)})")
    
    return results

# 步驟 3: 繪圖 (修改)
def plot_results(results):
    """Plot result charts"""
    if results is None: return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'BCI Classifier (1D-CNN - {Config.N_FEATURES_TOTAL} Features) - LOSO Results', fontsize=16)
    
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
    
    # 3. Training loss curves (現在 Keras 會提供數據)
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]): # 只畫前 5 個
            axes[2].plot(loss_curve, alpha=0.7, label=f'Fold {i+1}')
        axes[2].set_title('Training Loss Curves (First 5 Folds)')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss (binary_crossentropy)')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves available', 
                     horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Training Loss Curves')
    
    plt.tight_layout()
    plt.savefig('bci_results_1dcnn_bandpower.png', dpi=300, bbox_inches='tight')
    plt.show()

# 步驟 4: Main 函數 (修改)
def main():
    print("BCI EEG Classification (1D-CNN - Sequential Features) - Relaxation vs Concentration")
    print("=" * 70)
    
    # 設定 Keras/TF 的隨機種子以確保可重現性
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
    
    print(f"\nResults saved to 'bci_results_1dcnn_bandpower.png'")

if __name__ == "__main__":
    main()
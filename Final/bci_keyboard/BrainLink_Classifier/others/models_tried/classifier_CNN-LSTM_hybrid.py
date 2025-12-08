"""
Brain-Computer Interface CNN-LSTM Classifier
For EEG signal relaxation/focus state classification
Architecture: Conv1D -> MaxPooling -> LSTM -> Dense
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
    Input,
    Conv1D,             # CNN 層
    MaxPooling1D,       # 池化層 (降維)
    LSTM,               # LSTM 層
    Dense, 
    Dropout, 
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# === GPU Memory Setting ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)
# ==========================

class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_113-3"      # Train with junk-eliminated sets

    # === Band power setting ===
    BANDS = {
        'Delta': [1, 5],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [12, 30],
        'Gamma': [30, 45]
    }
    
    # === Feature list ===
    FEATURE_LIST = [
        'Theta',
        'Beta',
        # 頻域 (比值)
        'AB_Ratio',     # Alpha / Beta
        'TB_Ratio',
        'A_sum_ABT',    # α / (α + β + θ)
        
        # 頻域 (混亂度/位置)
        'Spec_Entropy',      # 頻譜熵
        
        # 時域 (複雜度)
        'Hjorth_Mobility',   # Hjorth 移動性
        'Hjorth_Complexity'  # Hjorth 複雜度
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST)
    
    # === Signal processing ===
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 2               
    OVERLAP_RATIO = 0.8                
    
    # === CNN-LSTM Hyperparameters ===
    # 
    # 1. 序列長度
    SEQUENCE_TIMESTEPS = 15      # !(可調) 模型一次看幾個窗口 (記憶長度)
    
    # 2. CNN 部分 (特徵提取)
    CNN_FILTERS = 64             # !(可調) 卷積核數量 (提取多少種局部特徵)
    CNN_KERNEL_SIZE = 3          # !(可調) 卷積核大小 (一次看 3 個時間步)
    CNN_POOL_SIZE = 2            # !(可調) 池化大小 (將序列長度縮小幾倍，例如 15 -> 7)
                                 # 這能減輕 LSTM 的負擔並提取顯著特徵
    
    # 3. LSTM 部分 (時序記憶)
    LSTM_UNITS = 64              # !(可調) LSTM 記憶單元數量
    LSTM_RECURRENT_DROPOUT = 0   # !For cuDNN acceleration 
    # LSTM_RECURRENT_DROPOUT = 0.5 # !(可調) 防止 LSTM 過擬合
    


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

# === Eliminate unwanted part ===
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

# === Load data ===
def load_eeg_data(subject_path):
    relax_file = os.path.join(subject_path, "1.txt"); focus_file = os.path.join(subject_path, "2.txt")
    try:
        relax_data = np.loadtxt(relax_file); focus_data = np.loadtxt(focus_file)
        relax_data = preprocess_signal(relax_data); focus_data = preprocess_signal(focus_data)
        return relax_data, focus_data
    except Exception as e:
        print(f"Error loading data for {subject_path}: {e}"); return None, None

# === Create segments ===
def create_segments(data, segment_length_samples, overlap_samples):
    if len(data) < segment_length_samples: return np.array([])
    segments = []; start = 0; step = segment_length_samples - overlap_samples
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]; segments.append(segment); start += step
    return np.array(segments)

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

# === Extract features ===
def extract_features(segments, fs, bands, feature_list):
    band_names_ordered = list(bands.keys()); all_features = []; EPSILON = 1e-10 
    for segment in segments:
        rel_powers, psd = get_band_power(segment, fs, bands)
        psd_norm = psd / (np.sum(psd) + EPSILON) 
        spec_entropy = entropy(psd_norm)
        d1 = np.diff(segment); d2 = np.diff(d1)      
        var_0 = np.var(segment); var_d1 = np.var(d1); var_d2 = np.var(d2)     
        hjorth_mobility = np.sqrt(var_d1 / (var_0 + EPSILON))
        mobility_d1 = np.sqrt(var_d2 / (var_d1 + EPSILON)) 
        hjorth_complexity = mobility_d1 / (hjorth_mobility + EPSILON)
        
        delta = rel_powers.get('Delta', 0); theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0); beta  = rel_powers.get('Beta', 0)
        gamma = rel_powers.get('Gamma', 0); feature_vector = []
        
        for feature_name in feature_list:
            if feature_name == 'Delta': feature_vector.append(delta)
            elif feature_name == 'Theta': feature_vector.append(theta)
            elif feature_name == 'Alpha': feature_vector.append(alpha)
            elif feature_name == 'Beta': feature_vector.append(beta)
            elif feature_name == 'Gamma': feature_vector.append(gamma)
            elif feature_name == 'A_sum_ABT': den = alpha + beta + theta; feature_vector.append(alpha / (den + EPSILON))
            elif feature_name == 'B_sum_AT': den = alpha + theta; feature_vector.append(beta / (den + EPSILON))
            elif feature_name == 'A_sum_BT': den = beta + theta; feature_vector.append(alpha / (den + EPSILON))
            elif feature_name == 'AB_Ratio': feature_vector.append(alpha / (beta + EPSILON))
            elif feature_name == 'TB_Ratio': feature_vector.append(theta / (beta + EPSILON))
            elif feature_name == 'Spec_Entropy': feature_vector.append(spec_entropy)
            elif feature_name == 'Hjorth_Mobility': feature_vector.append(hjorth_mobility)
            elif feature_name == 'Hjorth_Complexity': feature_vector.append(hjorth_complexity)
            else: warnings.warn(f"Unknown feature: {feature_name}"); feature_vector.append(0.0)
                
        all_features.append(feature_vector)
    return np.array(all_features)

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

# === CNN-LSTM Classifier ===
class EnhancedBCIClassifier:
    """Enhanced Brain-Computer Interface Classifier (CNN-LSTM Wrapper)"""
    
    def __init__(self):
        self.model = None; self.history = None; self.scaler = StandardScaler()
        self.feature_selector = None; self.threshold_ = 0.5
        self.smooth_k = 5; self.vote_k = 7

    # === Core architecture definition ===
    def build_model(self, n_features, n_timesteps):
        """建立 CNN-LSTM Keras 模型"""
        
        model = Sequential(name="BCI_CNN_LSTM")
        model.add(Input(shape=(n_timesteps, n_features), name="Input_Sequence"))
        
        # --- 1. CNN 部分 (特徵提取) ---
        # 提取局部時間模式
        model.add(Conv1D(
            filters=Config.CNN_FILTERS, 
            kernel_size=Config.CNN_KERNEL_SIZE,
            activation='relu',              # leaky_ReLU, ReLU or GeLU
            padding='same',
            name="Conv1D_Extractor"
        ))
        model.add(BatchNormalization())
        
        # --- 2. 池化層 (降維) ---
        # 縮減序列長度，減輕 LSTM 負擔
        model.add(MaxPooling1D(pool_size=Config.CNN_POOL_SIZE, name="MaxPooling"))
        
        # --- 3. LSTM 部分 (時序記憶) ---
        # 讀取經過 CNN 濃縮後的特徵序列
        model.add(LSTM(
            units=Config.LSTM_UNITS,
            recurrent_dropout=Config.LSTM_RECURRENT_DROPOUT,
            return_sequences=False, # 只輸出最後狀態
            name="LSTM_Layer"
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.5)) # 補償 Dropout
        
        # --- 4. 全連接層 ---
        model.add(Dense(Config.DENSE_UNITS, activation='relu', name="Dense_1"))
        model.add(Dropout(Config.DROPOUT))
        
        # --- 5. 輸出層 ---
        model.add(Dense(1, activation='sigmoid', name="Output_Sigmoid"))
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            metrics=['accuracy']
        )
        return model

    def fit(self, X_seq, y_seq):
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        X_selected = X_scaled_seq

        tf.keras.backend.clear_session()
        self.model = self.build_model(n_features, n_timesteps)
        
        early_stopping_cb = EarlyStopping(
            monitor='val_loss', 
            patience=Config.EARLY_STOP_PATIENCE,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_selected, y_seq,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_FRACTION, 
            callbacks=[early_stopping_cb],
            verbose=0
        )
        
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
            print(f"Warning: Failed to find threshold: {e}"); self.threshold_ = 0.5
        return self

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
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled_seq = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        X_selected = X_scaled_seq
        raw_probabilities = self.model.predict(X_selected).ravel()
        return self._postprocess_predict(raw_probabilities) 

    def get_loss_curve(self):
        if self.history and 'loss' in self.history.history: return self.history.history['loss']
        return []

# === Main function ===
def load_all_segments_and_features():
    all_features = []; all_labels = []; all_subjects = []
    subject_folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    if not subject_folders: print(f"Error: No subject folders found in {Config.DATASET_PATH}"); return None, None, None
    
    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_data, focus_data = load_eeg_data(subject_folder)
        if relax_data is None or focus_data is None: continue

        segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
        overlap_samples = int(Config.OVERLAP_RATIO * segment_length_samples) # Fixed bug
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
    return np.vstack(all_features), np.hstack(all_labels), np.array(all_subjects)

def leave_one_subject_out_validation():
    print(f"Starting LOSO Cross-Validation (CNN-LSTM)...")
    print(f"Features: {Config.FEATURE_LIST}")
    print(f"Sequence: {Config.SEQUENCE_TIMESTEPS} steps, CNN Filters: {Config.CNN_FILTERS}, LSTM Units: {Config.LSTM_UNITS}")
    
    X_seg, y_seg, subjects_array = load_all_segments_and_features()
    if X_seg is None: return None
    unique_subjects = sorted(list(set(np.unique(subjects_array))))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    for test_subject in unique_subjects:
        train_mask = (subjects_array != test_subject); test_mask = (subjects_array == test_subject)
        X_train_seg, y_train_seg, subjects_train = X_seg[train_mask], y_seg[train_mask], subjects_array[train_mask]
        X_test_seg, y_test_seg, subjects_test = X_seg[test_mask], y_seg[test_mask], subjects_array[test_mask]
        
        X_train, y_train = create_sequences_from_groups(X_train_seg, y_train_seg, subjects_train, Config.SEQUENCE_TIMESTEPS)
        X_test, y_test = create_sequences_from_groups(X_test_seg, y_test_seg, subjects_test, Config.SEQUENCE_TIMESTEPS)
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping {test_subject}: Not enough data."); continue
            
        classifier = EnhancedBCIClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        results['accuracies'].append(accuracy); results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve()); results['subject_names'].append(test_subject)
        print(f"{test_subject}: Accuracy = {accuracy:.3f} (N_train={len(X_train)}, N_test={len(X_test)})")
    return results

def plot_results(results):
    if results is None: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'BCI Classifier (CNN-LSTM) - LOSO Results', fontsize=16)
    
    axes[0].bar(range(len(results['accuracies'])), results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.6 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject'); axes[0].set_xlabel('Subject Index'); axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1)
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Relax', 'Focus'], yticklabels=['Relax', 'Focus'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f'Fold {i+1}')
        axes[2].set_title('Training Loss Curves (First 5 Folds)'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves', transform=axes[2].transAxes)
        axes[2].set_title('Training Loss Curves')
    
    plt.tight_layout(); plt.savefig('bci_results_cnn-lstm_no_junk.png', dpi=300, bbox_inches='tight'); plt.show()

def main():
    print("BCI EEG Classification (CNN-LSTM) - Relaxation vs Concentration (Poor-quality sets removed)"); print("=" * 70)
    tf.random.set_seed(Config.RANDOM_STATE); np.random.seed(Config.RANDOM_STATE)
    
    results = leave_one_subject_out_validation()
    if results is None: print("Validation failed!"); return
    
    mean_accuracy = np.mean(results['accuracies']); std_accuracy = np.std(results['accuracies'])
    print(f"\nOverall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_acc = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        focus_acc = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
    print(f"\nRelax Accuracy: {relax_acc:.3f}"); print(f"Concentration Accuracy: {focus_acc:.3f}")
    plot_results(results); print(f"\nResults saved to 'bci_results_cnn-lstm_no_junk.png'")

if __name__ == "__main__":
    main()
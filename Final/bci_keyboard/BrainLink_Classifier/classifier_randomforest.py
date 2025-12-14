

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
# === CHANGED (Route B): 新增 RandomForest ===
from sklearn.ensemble import RandomForestClassifier

import os
import glob
import warnings
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')


class Config:
    # Dataset path settings
    DATASET_PATH = r"C:\Users\袁子堯\Downloads\bci_dataset_113-2\bci_dataset_113-2"
    #DATASET_PATH = r"C:\Users\袁子堯\Downloads\bci_dataset_113-2 _test\bci_dataset_113-2"

    # === Band-Power Feature Configuration ===
    BANDS = {
        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [13, 30],
        'Gamma': [30, 45]
    }

    #    #
    FEATURE_LIST = [
        'Theta',        # Basic Relative Power
        'Beta',         # Basic Relative Power
        'AB_Ratio',     # Alpha / Beta
        'A_sum_ABT',    # α / (α + β + θ)
        #'B_sum_AT',     # β / (α + θ)
        #'A_sum_BT',     # α / (β + θ)
        'TB_Ratio',     # θ / β 
        'Spec_Entropy',      # Spectral Entropy
        'Hjorth_Mobility',   # Hjorth Mobility
        'Hjorth_Complexity'  # Hjorth Complexity
    ]
    N_FEATURES_TOTAL = len(FEATURE_LIST)  # Auto compute

    # === Signal processing ===
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 3
    OVERLAP_RATIO = 0.8

    # === segment setting ===
    SEQUENCE_TIMESTEPS = 15


    # === Add random forest hyperparameters ===

    RF_N_ESTIMATORS = 1000        # Number of Trees
    RF_MAX_DEPTH = 10             # Maximum Depth
    RF_MIN_SAMPLES_LEAF = 10      # Minimum Samples per Leaf
    RF_N_JOBS = -1                # all CPU core
    RF_RANDOM_STATE = 42
    RF_MAX_FEATURES = "sqrt"      # Maximum Features

    # Feature selection 
    FEATURE_SELECTION = False
    N_FEATURES_SELECT = N_FEATURES_TOTAL

    # Other settings
    RANDOM_STATE = 42


# === student preprocessing ===
def preprocess_signal(data):
    fs = 500
    seg_len = int(round(20.0 * fs))
    gap_len = int(round(20.0 * fs))
    start = int(round(20.0 * fs))
    max_end = min(len(data), int(round(400.0 * fs)))
    valid_idx = []
    while start + seg_len <= max_end:
        valid_idx.append((start, start + seg_len))
        start += seg_len + gap_len
    if not valid_idx:
        x = np.asarray(data, dtype=float)
    else:
        x = np.concatenate([data[s:e] for s, e in valid_idx], axis=0).astype(float)
    x = x - np.median(x)
    dc_window_sec = 2.0
    w = int(round(dc_window_sec * fs))
    if w >= 2 and len(x) > w:
        kernel = np.ones(w, dtype=float) / float(w)
        baseline = np.convolve(x, kernel, mode='same')
        x = x - baseline
    return x


# === load_eeg_data ===
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
        print(f"Error loading data for {subject_path}: {e}")
        return None, None


# === create_segments ===
def create_segments(data, segment_length_samples, overlap_samples):
    if len(data) < segment_length_samples:
        return np.array([])
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        segments.append(segment)
        start += step
    return np.array(segments)


# === Basic Band-Power Function ===
def get_band_power(epoch_data, fs, bands):
    freqs, psd = signal.welch(epoch_data, fs, nperseg=len(epoch_data), window='hann')
    freq_res = freqs[1] - freqs[0]
    abs_powers = {}
    for band_name, (low_hz, high_hz) in bands.items():
        idx_band = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
        if len(idx_band) == 0:
            abs_powers[band_name] = 0
            continue
        power = simpson(psd[idx_band], dx=freq_res)
        abs_powers[band_name] = power
    total_power = sum(abs_powers.values())
    if total_power == 0:
        return {band: 0 for band in bands}, psd
    rel_powers = {band: power / total_power for band, power in abs_powers.items()}
    return rel_powers, psd


# === Dynamic Feature Extraction Function ===
def extract_features(segments, fs, bands, feature_list):
    band_names_ordered = list(bands.keys())
    all_features = []
    EPSILON = 1e-10

    for segment in segments:
        # --- Compute Band-Power Features---
        rel_powers, psd = get_band_power(segment, fs, bands)

        # --- Compute new Features ---
        #  (Spectral Entropy)
        psd_norm = psd / (np.sum(psd) + EPSILON)
        spec_entropy = entropy(psd_norm)

        #  Hjorth Hyperparameters
        d1 = np.diff(segment)  
        d2 = np.diff(d1)       

        var_0 = np.var(segment)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)

        hjorth_mobility = np.sqrt(var_d1 / (var_0 + EPSILON))
        mobility_d1 = np.sqrt(var_d2 / (var_d1 + EPSILON))
        hjorth_complexity = mobility_d1 / (hjorth_mobility + EPSILON)

        # ---  Retrieve Basic Band Values ---
        delta = rel_powers.get('Delta', 0)
        theta = rel_powers.get('Theta', 0)
        alpha = rel_powers.get('Alpha', 0)
        beta = rel_powers.get('Beta', 0)
        gamma = rel_powers.get('Gamma', 0)
        feature_vector = []

        # --- Construct the Feature Vector Based on the Feature_List ---
        for feature_name in feature_list:
            if feature_name == 'Delta':
                feature_vector.append(np.log10(delta))
            elif feature_name == 'Theta':
                feature_vector.append(np.log10(theta))
            elif feature_name == 'Alpha':
                feature_vector.append(np.log10(alpha))
            elif feature_name == 'Beta':
                feature_vector.append(np.log10(beta))
            elif feature_name == 'A_sum_ABT':
                den = alpha + beta + theta
                feature_vector.append(np.log10(alpha / (den + EPSILON)))
            elif feature_name == 'B_sum_AT':
                den = alpha + theta
                feature_vector.append(np.log10(beta / (den + EPSILON)))
            elif feature_name == 'TB_Ratio':
                feature_vector.append(np.log10(theta / (beta + EPSILON)))
            elif feature_name == 'A_sum_BT':
                den = beta + theta
                feature_vector.append(np.log10(alpha / (den + EPSILON)))
            elif feature_name == 'AB_Ratio':
                feature_vector.append(np.log10(alpha / (beta + EPSILON)))
            elif feature_name == 'Spec_Entropy':
                feature_vector.append(spec_entropy)
            elif feature_name == 'Hjorth_Mobility':
                feature_vector.append(hjorth_mobility)
            elif feature_name == 'Hjorth_Complexity':
                feature_vector.append(hjorth_complexity)
            else:
                warnings.warn(f"Unknown feature: {feature_name}")
                feature_vector.append(0.0)

        all_features.append(feature_vector)

    return np.array(all_features)


# ===  Keras Data Preparation Function (retained for sequence construction but later flattened for RF) ===
def create_sequences_from_groups(X_seg, y_seg, subjects_array, timesteps):
    X_seq_list = []
    y_seq_list = []
    unique_subjects = np.unique(subjects_array)

    for subject in unique_subjects:
        subject_mask = (subjects_array == subject)
        X_subject = X_seg[subject_mask]
        y_subject = y_seg[subject_mask]

        if len(X_subject) < timesteps:
            continue

        for i in range(len(X_subject) - timesteps + 1):
            X_seq_list.append(X_subject[i: i + timesteps])
            y_seq_list.append(y_subject[i + timesteps - 1])

    if not X_seq_list:
        return np.array([]), np.array([])

    return np.array(X_seq_list), np.array(y_seq_list)


# === Classical ML-Based BCI Classifier (RandomForest + StandardScaler) ===
class ClassicalBCIClassifier:
    """Classical Brain-Computer Interface Classifier (RandomForest Wrapper)"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.loss_curve_ = []  # 保留接口

    def fit(self, X_seq, y_seq):
        """
        train RandomForest classifier
        X_seq: (n_samples, n_timesteps, n_features)
        y_seq: (n_samples,)
        """
        n_samples, n_timesteps, n_features = X_seq.shape

        # Flatten the sequence into a one-dimensional feature vector: [t1_features, t2_features, ..., t_T_features]
        X_flat = X_seq.reshape(n_samples, n_timesteps * n_features)

        # Scaling
        X_scaled = self.scaler.fit_transform(X_flat)

        # Build RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            max_features=Config.RF_MAX_FEATURES,      
            class_weight="balanced_subsample",         
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RF_RANDOM_STATE
        )

        self.model.fit(X_scaled, y_seq)
        self.loss_curve_ = []
        return self

    def predict(self, X_seq):
        """
        Use RandomForest to predict
        X_seq: (n_samples, n_timesteps, n_features)
        return: (n_samples,) 0/1 label
        """
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(n_samples, n_timesteps * n_features)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def get_loss_curve(self):
        """ Return an empty list to maintain compatibility with the original interface"""
        return self.loss_curve_


# === Main Execution Function ===

# step 1: load all "segments"
def load_all_segments_and_features():
    all_features = []
    all_labels = []
    all_subjects = []
    subject_folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    if not subject_folders:
        print(f"Error: No subject folders found in {Config.DATASET_PATH}")
        return None, None, None

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_data, focus_data = load_eeg_data(subject_folder)
        if relax_data is None or focus_data is None:
            continue

        segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
        overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)
        relax_segments = create_segments(relax_data, segment_length_samples, overlap_samples)
        focus_segments = create_segments(focus_data, segment_length_samples, overlap_samples)

        if relax_segments.size == 0 or focus_segments.size == 0:
            print(f"Warning: Not enough data for {subject_id}. Skipping...")
            continue

        relax_features = extract_features(relax_segments, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)
        focus_features = extract_features(focus_segments, Config.SAMPLING_RATE, Config.BANDS, Config.FEATURE_LIST)

        relax_labels = np.zeros(len(relax_features))
        focus_labels = np.ones(len(focus_features))
        subject_features = np.vstack([relax_features, focus_features])
        subject_labels = np.hstack([relax_labels, focus_labels])
        subject_ids = [subject_id] * len(subject_labels)

        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend(subject_ids)

    if not all_features:
        print("Error: No valid data found")
        return None, None, None

    X_seg = np.vstack(all_features)
    y_seg = np.hstack(all_labels)
    subjects_array = np.array(all_subjects)
    return X_seg, y_seg, subjects_array


# step 2: LOSO Validation
def leave_one_subject_out_validation():
    print(f"Starting LOSO Cross-Validation (RandomForest, flattened sequences) with {Config.N_FEATURES_TOTAL} features...")
    print(f"Features: {Config.FEATURE_LIST}")
    print(f"Sequence: {Config.SEQUENCE_TIMESTEPS} timesteps (flattened), Segment: {Config.SEGMENT_LENGTH}s, Overlap: {Config.OVERLAP_RATIO}")

    X_seg, y_seg, subjects_array = load_all_segments_and_features()
    if X_seg is None:
        return None

    unique_subjects = sorted(list(set(np.unique(subjects_array))))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}

    for test_subject in unique_subjects:
        train_mask = (subjects_array != test_subject)
        test_mask = (subjects_array == test_subject)
        X_train_seg, y_train_seg, subjects_train = X_seg[train_mask], y_seg[train_mask], subjects_array[train_mask]
        X_test_seg, y_test_seg, subjects_test = X_seg[test_mask], y_seg[test_mask], subjects_array[test_mask]

        X_train, y_train = create_sequences_from_groups(X_train_seg, y_train_seg, subjects_train, Config.SEQUENCE_TIMESTEPS)
        X_test, y_test = create_sequences_from_groups(X_test_seg, y_test_seg, subjects_test, Config.SEQUENCE_TIMESTEPS)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping {test_subject}: Not enough data after creating sequences.")
            continue

        classifier = ClassicalBCIClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        results['accuracies'].append(accuracy)
        results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve())
        results['subject_names'].append(test_subject)

        print(f"{test_subject}: Accuracy = {accuracy:.3f} (N_train={len(X_train)}, N_test={len(X_test)})")

    return results


# step3 plot figure
def plot_results(results):
    if results is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'BCI Classifier (RandomForest - {Config.N_FEATURES_TOTAL} Features) - LOSO Results',
                 fontsize=16)

    # 1. Accuracy
    axes[0].bar(range(len(results['accuracies'])), results['accuracies'],
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.6 else 'red'
                       for acc in results['accuracies']])
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
                xticklabels=['Relax', 'Focus'], yticklabels=['Relax', 'Focus'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    # 3. Training loss curves (RandimForest doesm't have loss curve, empty)
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f'Fold {i + 1}')
        axes[2].set_title('Training Loss Curves (First 5 Folds)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves available (RandomForest)',
                     horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Training Loss Curves')

    plt.tight_layout()
    plt.savefig('bci_results_rf_bandpower.png', dpi=300, bbox_inches='tight')
    plt.show()


# step 4: Main function
def main():
    print("BCI EEG Classification (RandomForest - Sequential Features Flattened) - Relaxation vs Concentration")
    print("=" * 70)

    np.random.seed(Config.RANDOM_STATE)

    results = leave_one_subject_out_validation()
    if results is None:
        print("Validation failed!")
        return

    mean_accuracy = np.mean(results['accuracies'])
    std_accuracy = np.std(results['accuracies'])
    print(f"\nOverall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")

    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        concentration_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        concentration_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0

    print(f"\nRelax Class:")
    print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")

    print(f"\nConcentration Class:")
    print(f"  - Accuracy (Recall): {concentration_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {concentration_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")

    plot_results(results)
    print(f"\nResults saved to 'bci_results_rf_bandpower.png'")


if __name__ == "__main__":
    main()

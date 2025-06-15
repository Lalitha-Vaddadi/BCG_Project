import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis

base_path = 'C:/Users/lalit/Downloads/dataset/data'
fs = 100
win_size = 30 * fs
subjects_to_use = {'01', '02', '03'}

def butter_bandpass_filter(data, lowcut=0.5, highcut=10.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def extract_features(segment):
    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'rms': np.sqrt(np.mean(segment**2)),
        'range': np.ptp(segment),
        'skewness': skew(segment),
        'kurtosis': kurtosis(segment),
        'energy': np.sum(segment**2)
    }

def preprocess_extract():
    for subject_id in sorted(os.listdir(base_path)):
        if subject_id not in subjects_to_use:
            continue

        bcg_path = os.path.join(base_path, subject_id, 'BCG')
        feat_path = os.path.join(base_path, subject_id, 'features')
        os.makedirs(feat_path, exist_ok=True)

        for file in sorted(os.listdir(bcg_path)):
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(bcg_path, file)
            df = pd.read_csv(file_path)
            signal = df.iloc[:, 0].dropna().values
            filtered = butter_bandpass_filter(signal)

            feats = []
            for i in range(0, len(filtered) - win_size, win_size):
                seg = filtered[i:i + win_size]
                f = extract_features(seg)
                f['segment'] = i // win_size
                feats.append(f)

            if feats:
                feature_array = pd.DataFrame(feats).to_numpy()
                out_file = os.path.join(feat_path, f"{file.replace('.csv', '')}_features.npy")
                np.save(out_file, feature_array)

if __name__ == "__main__":
    preprocess_extract()

import os
import numpy as np
import pandas as pd

base_path = 'C:/Users/lalit/Downloads/dataset/data'
subjects_to_use = {'01', '02', '03'}

def display_npy_features():
    for subject_id in sorted(os.listdir(base_path)):
        if subject_id not in subjects_to_use:
            continue

        feat_path = os.path.join(base_path, subject_id, 'features')
        if not os.path.isdir(feat_path):
            continue

        print(f"\n👤 Subject {subject_id}")
        for file in sorted(os.listdir(feat_path)):
            if file.endswith('.npy'):
                file_path = os.path.join(feat_path, file)
                features = np.load(file_path)
                df = pd.DataFrame(features, columns=['mean', 'std', 'rms', 'range', 'skewness', 'kurtosis', 'energy', 'segment'])
                print(f"\n📄 {file}")
                print(df.head())

if __name__ == "__main__":
    display_npy_features()

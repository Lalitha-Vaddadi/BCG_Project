import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_path = 'C:/Users/lalit/Downloads/dataset/data'

def plot_relationships(subject_id='01'):
    subject_path = os.path.join(base_path, subject_id, 'features')
    if not os.path.exists(subject_path):
        print(f"No features found for Subject {subject_id}")
        return

    all_features = []
    for file in sorted(os.listdir(subject_path)):
        if file.endswith('.npy'):
            arr = np.load(os.path.join(subject_path, file))
            all_features.append(pd.DataFrame(arr, columns=['mean', 'std', 'rms', 'range', 'skewness', 'kurtosis', 'energy', 'segment']))

    if not all_features:
        print("No .npy feature files found.")
        return

    one_night = all_features[0]
    all_nights = pd.concat(all_features, ignore_index=True)

    for label, df in [("One Night", one_night), ("All Nights", all_nights)]:
        print(f"\n📊 Plotting for {label}")
        feat_df = df.drop(columns=['segment'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(feat_df.corr(), annot=True, cmap='coolwarm')
        plt.title(f"Correlation Matrix - {label}")
        plt.tight_layout()
        plt.show()

        sns.pairplot(feat_df)
        plt.suptitle(f"Pair Plot - {label}", y=1.02)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_relationships(subject_id='01')

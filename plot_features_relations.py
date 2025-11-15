import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (Must match Preprocessing) ---
BASE_PATH = 'C:/Users/lalit/Downloads/dataset/data'
FS = 140  # Sampling Rate (Hz)
WIN_DURATION_SEC = 30
WIN_SIZE = WIN_DURATION_SEC * FS  # 4200 samples

# PROCESS ONLY THE FIRST 2 SUBJECTS FOR FEATURE EXTRACTION AND EDA
# NOTE: The original prompt requested subjects '01' and '02' for the FINAL script processing limit.
SUBJECTS_TO_PROCESS = ['01', '02']
EDA_SUBJECT = '01'  # Subject to use for detailed EDA plots


# --- Feature Calculation ---
def calculate_features(segment):
    """Calculates time-domain features for a single, clean BCG segment."""
    # Ensure segment is not empty (it shouldn't be if .dropna() was used)
    if segment.empty:
        return np.array([np.nan] * 7)

    # Convert to NumPy array for calculation
    data = segment.values

    # 1. Mean (Average amplitude)
    mean_val = np.mean(data)
    # 2. Standard Deviation (Variability)
    std_val = np.std(data)
    # 3. Root Mean Square (RMS - Signal power)
    rms_val = np.sqrt(np.mean(data ** 2))
    # 4. Range (Peak-to-Peak amplitude)
    range_val = np.max(data) - np.min(data)
    # 5. Skewness (Symmetry of the distribution)
    skewness_val = skew(data)
    # 6. Kurtosis (Heaviness of the tails / peakedness)
    kurtosis_val = kurtosis(data)
    # 7. Energy (Sum of squares - Total power)
    energy_val = np.sum(data ** 2)

    return np.array([mean_val, std_val, rms_val, range_val, skewness_val, kurtosis_val, energy_val])


# --- Main Feature Extraction Loop ---
def run_feature_extraction():
    # Identify all subject folders
    subject_ids = sorted(
        [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and d.isdigit()])

    # Filter to only run for the specified subjects
    subject_ids = [s for s in subject_ids if s in SUBJECTS_TO_PROCESS]

    for subject_id in subject_ids:
        preprocessed_path = os.path.join(BASE_PATH, subject_id, 'preprocessed_data')
        features_path = os.path.join(BASE_PATH, subject_id, 'features')
        os.makedirs(features_path, exist_ok=True)

        if not os.path.isdir(preprocessed_path):
            print(f"Skipping Subject {subject_id}: 'preprocessed_data' folder not found. Run preprocessing first.")
            continue

        print(f"\nExtracting features for Subject {subject_id}...")

        # Process ALL preprocessed CSV files for this subject
        preprocessed_files = sorted([f for f in os.listdir(preprocessed_path) if f.endswith('_preprocessed_BCG.csv')])

        for file in preprocessed_files:
            file_path = os.path.join(preprocessed_path, file)
            df = pd.read_csv(file_path)

            # Get the preprocessed BCG signal column
            bcg_signal = df['BCG_Preprocessed']

            all_features = []

            # Segment the data and process each epoch
            for i in range(0, len(bcg_signal), WIN_SIZE):
                segment = bcg_signal.iloc[i:i + WIN_SIZE]

                # Check if the segment is full length
                if len(segment) == WIN_SIZE:
                    # Crucial step: Select only non-NaN epochs
                    clean_segment = segment.dropna()

                    # If the segment is full length AND contains no NaNs (i.e., it was a 'clean' segment)
                    if len(clean_segment) == WIN_SIZE:
                        features = calculate_features(clean_segment)
                        all_features.append(features)

            # Convert list of feature arrays into a final NumPy array
            if all_features:
                feature_matrix = np.array(all_features)

                # --- Data Saving (NumPy Format) ---
                # Example file: 01_20231104_preprocessed_BCG.csv
                # Get the date part: 20231104
                file_date_part = file.split('_')[1]
                out_file_name = f"{subject_id}_{file_date_part}_BCG_features.npy"
                out_file_path = os.path.join(features_path, out_file_name)

                np.save(out_file_path, feature_matrix)
                print(
                    f"  Features saved for {file_date_part} ({feature_matrix.shape[0]} epochs) to: {subject_id}/features/{out_file_name}")
            else:
                print(f"  No clean epochs found for {file}. Skipping feature saving.")


# --- Exploratory Data Analysis (EDA) ---
def run_eda():
    subject_id = EDA_SUBJECT
    features_path = os.path.join(BASE_PATH, subject_id, 'features')
    eda_viz_path = os.path.join(BASE_PATH, subject_id, 'eda_plots')
    os.makedirs(eda_viz_path, exist_ok=True)

    if not os.path.isdir(features_path):
        print(f"\nSkipping EDA: Feature folder not found for Subject {subject_id}.")
        return

    print(f"\nRunning EDA for Subject {subject_id}...")

    feature_files = sorted([f for f in os.listdir(features_path) if f.endswith('_BCG_features.npy')])

    if not feature_files:
        print(f"  No feature files found for EDA in {subject_id}/features.")
        return

    # Define feature names for columns
    FEATURE_NAMES = ['Mean', 'Std', 'RMS', 'Range', 'Skewness', 'Kurtosis', 'Energy']

    # --- 1. Within-Night Visualization (First Night) ---
    first_file = feature_files[0]
    first_file_path = os.path.join(features_path, first_file)
    single_night_features = np.load(first_file_path)
    df_single = pd.DataFrame(single_night_features, columns=FEATURE_NAMES)

    # Get date for plot title
    date_part = first_file.split('_')[1]

    # a. Correlation Heatmap (Single Night)
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_single.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Feature Correlation Heatmap (Sub {subject_id}, Night {date_part})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_viz_path, f'{date_part}_Correlation_Heatmap.png'))
    plt.close()
    print(f"  Saved Correlation Heatmap for single night: {date_part}_Correlation_Heatmap.png")

    # b. Pair Plot (Single Night) - Shows distribution and bivariate relationships
    pair_plot_single = sns.pairplot(df_single, plot_kws={'alpha': 0.6, 's': 5})
    pair_plot_single.fig.suptitle(f'Feature Pair Plot (Sub {subject_id}, Night {date_part})', y=1.02)
    pair_plot_single.fig.set_size_inches(14, 14)
    plt.savefig(os.path.join(eda_viz_path, f'{date_part}_Pair_Plot.png'))
    plt.close()
    print(f"  Saved Pair Plot for single night: {date_part}_Pair_Plot.png")

    # --- 2. Across-All-Nights Visualization (Concatenate all feature files) ---
    all_nights_features = []

    # Store night/session label for multi-session plots
    session_labels = []

    for f_name in feature_files:
        f_path = os.path.join(features_path, f_name)
        features = np.load(f_path)
        all_nights_features.append(features)

        # Get date for labeling
        date_label = f_name.split('_')[1]
        session_labels.extend([date_label] * features.shape[0])

    if all_nights_features:
        full_feature_matrix = np.vstack(all_nights_features)
        df_all = pd.DataFrame(full_feature_matrix, columns=FEATURE_NAMES)
        df_all['Night'] = session_labels

        # c. Correlation Heatmap (All Nights)
        plt.figure(figsize=(8, 7))
        sns.heatmap(df_all.drop(columns='Night').corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Feature Correlation Heatmap (Sub {subject_id}, All Nights)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_viz_path, 'All_Nights_Correlation_Heatmap.png'))
        plt.close()
        print("  Saved Correlation Heatmap across all nights.")

        # d. Pair Plot (All Nights - with Night as Hue)
        pair_plot_all = sns.pairplot(df_all, hue='Night', plot_kws={'alpha': 0.6, 's': 5})
        pair_plot_all.fig.suptitle(f'Feature Pair Plot (Sub {subject_id}, All Nights by Session)', y=1.02)
        pair_plot_all.fig.set_size_inches(14, 14)
        plt.savefig(os.path.join(eda_viz_path, 'All_Nights_Pair_Plot.png'))
        plt.close()
        print("  Saved Pair Plot across all nights (color-coded by night).")


if __name__ == "__main__":
    # 1. Run Feature Extraction for Subjects '01' and '02'
    run_feature_extraction()

    # 2. Run EDA for Subject '01'
    run_eda()
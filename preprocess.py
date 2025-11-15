import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# --- Configuration ---
# NOTE: Update this path to your dataset location
BASE_PATH = 'C:/Users/lalit/Downloads/dataset/data'
FS = 140  # Corrected Sampling Rate (Hz)
WIN_DURATION_SEC = 30
WIN_SIZE = WIN_DURATION_SEC * FS  # 4200 samples
CARDIAC_LOWCUT = 0.7
CARDIAC_HIGHCUT = 10.0  # Bandpass for cardiac isolation and analysis
# Noise Threshold: Segments with STD > THRESHOLD * MEDIAN(STD) are considered noise
STD_THRESHOLD_FACTOR = 3
FILTER_ORDER = 3

# --- Subject Filtering ---
# PROCESS ONLY THE FIRST 3 SUBJECTS TO SAVE TIME, as requested.
SUBJECTS_TO_PROCESS = ['01', '02', '03']


# --- Filter Implementation ---
def butter_bandpass_filter(data, lowcut, highcut, order=FILTER_ORDER):
    """Applies a Butterworth bandpass filter for cardiac isolation (0.7-10 Hz)."""
    nyq = 0.5 * FS
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# --- Normalization and Noise Removal ---
def standardize_signal(signal):
    """Standardizes the signal (Zero mean, unit variance)."""
    return (signal - np.mean(signal)) / np.std(signal)


def reject_noise(segments, std_threshold_factor):
    """
    Identifies indices of noisy segments based on standard deviation.
    Returns a list of boolean flags indicating if a segment is clean (True) or noisy (False).
    """
    if not segments:
        return [], []

    segment_stds = np.array([np.std(seg) for seg in segments])
    median_std = np.median(segment_stds)
    noise_threshold = median_std * std_threshold_factor

    # Boolean mask: True if clean, False if noisy
    is_clean_mask = segment_stds < noise_threshold

    return is_clean_mask


# --- Main Processing and Visualization Loop ---
def preprocess_and_visualize():
    # Identify all subject folders (assuming numbered folders 01, 02, etc.)
    subject_ids = sorted(
        [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and d.isdigit()])

    # Filter to only run for the specified subjects
    subject_ids = [s for s in subject_ids if s in SUBJECTS_TO_PROCESS]

    for subject_id in subject_ids:
        bcg_path = os.path.join(BASE_PATH, subject_id, 'BCG')
        if not os.path.isdir(bcg_path):
            print(f"Skipping Subject {subject_id}: BCG folder not found.")
            continue

        # --- Define output paths and create directories ---
        processed_data_path = os.path.join(BASE_PATH, subject_id, 'preprocessed_data')
        viz_path = os.path.join(BASE_PATH, subject_id, 'comparative_visualization')
        os.makedirs(processed_data_path, exist_ok=True)
        os.makedirs(viz_path, exist_ok=True)

        print(f"\nProcessing Subject {subject_id}...")

        # Process ALL BCG files for this subject
        bcg_files = sorted([f for f in os.listdir(bcg_path) if f.endswith('.csv')])

        for file in bcg_files:
            file_path = os.path.join(bcg_path, file)
            df = pd.read_csv(file_path)

            # --- Preprocessing Steps ---
            raw_signal = df.iloc[:, 0].dropna().values
            normalized_signal = standardize_signal(raw_signal)
            cardiac_filtered_signal = butter_bandpass_filter(normalized_signal, CARDIAC_LOWCUT, CARDIAC_HIGHCUT)

            # 3. Segmentation (for noise check)
            segments = []
            for i in range(0, len(cardiac_filtered_signal), WIN_SIZE):
                segment = cardiac_filtered_signal[i:i + WIN_SIZE]
                if len(segment) == WIN_SIZE:
                    segments.append(segment)

            if not segments:
                print(f"  Skipping {file}: Signal too short for segmentation.")
                continue

            # 4. Artifact Removal - Get mask
            is_clean_mask = reject_noise(segments, STD_THRESHOLD_FACTOR)

            # --- Signal Reconstruction and NaN-masking ---
            reconstructed_signal = np.copy(cardiac_filtered_signal)
            total_segments = len(is_clean_mask)
            noise_indices = []

            for i in range(total_segments):
                if not is_clean_mask[i]:
                    start = i * WIN_SIZE
                    end = start + WIN_SIZE
                    # Replace noisy segments with NaN marker
                    reconstructed_signal[start:end] = np.nan
                    noise_indices.append(i)

            # --- Data Saving (CSV Format) ---
            file_parts = file.split('_')
            if len(file_parts) >= 3:
                file_date_part = file_parts[-2]
            else:
                file_date_part = 'NODATE'

            out_file_name = f"{subject_id}_{file_date_part}_preprocessed_BCG.csv"
            out_file_path = os.path.join(processed_data_path, out_file_name)

            output_df = pd.DataFrame({
                'BCG_Preprocessed': reconstructed_signal,
                'Timestamp': df['Timestamp'].iloc[
                    :len(reconstructed_signal)].values if 'Timestamp' in df.columns else np.arange(
                    len(reconstructed_signal)) / FS,
                'fs': FS
            })

            output_df.to_csv(out_file_path, index=False)
            print(f"  Processed data saved for {file} to: {os.path.basename(processed_data_path)}/{out_file_name}")

            # --- Visualization for CURRENT File (All Nights) ---

            # Choose a time window for visualization (e.g., 5 minutes = 10 segments)
            viz_duration_sec = 5 * 60
            viz_length = viz_duration_sec * FS

            # Adjust viz length if file is shorter than 5 minutes
            if viz_length > len(raw_signal):
                viz_length = len(raw_signal)

            # Only proceed if there is enough data for visualization
            if viz_length > 0:
                time_axis = np.arange(0, viz_length / FS, 1 / FS)

                # Extract visualization snippets
                raw_snippet = standardize_signal(raw_signal[:viz_length])
                filtered_snippet = cardiac_filtered_signal[:viz_length]

                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                # Panel 1: Normalized Raw Signal
                axes[0].plot(time_axis, raw_snippet, color='#1f77b4', linewidth=1, label='Normalized Raw BCG')
                axes[0].set_title('1. Normalized Raw BCG Signal (Before Filtering and Noise Removal)', fontsize=12)
                axes[0].set_ylabel('Amplitude (Norm.)')
                axes[0].grid(True, alpha=0.5)
                axes[0].legend(loc='upper right')

                # Panel 2: Filtered Signal with Noise Highlights
                axes[1].plot(time_axis, filtered_snippet, color='#2ca02c', linewidth=1,
                             label=f'Filtered ({CARDIAC_LOWCUT}-{CARDIAC_HIGHCUT} Hz)')
                axes[1].set_title('2. Filtered Signal with Artifact Rejection (Cardiac Isolated)', fontsize=12)
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Amplitude (Norm.)')
                axes[1].grid(True, alpha=0.5)

                # Highlight rejected segments (only within the visualization window)
                rejected_in_viz = [i for i in noise_indices if (i * WIN_SIZE) < viz_length]

                for i, rejected_idx in enumerate(rejected_in_viz):
                    start_time = rejected_idx * WIN_DURATION_SEC
                    end_time = start_time + WIN_DURATION_SEC

                    # Add the span and label it only once
                    axes[1].axvspan(start_time, end_time, color='red', alpha=0.3,
                                    label='Rejected Segment' if i == 0 else None)

                if rejected_in_viz:
                    axes[1].legend(loc='upper right')

                plt.suptitle(
                    f'Subject {subject_id} - Preprocessing Verification ({viz_duration_sec // 60} Minutes of {file})',
                    fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Save the comparative plot to the subject's new folder
                # Filename fix: Use the date part directly
                viz_file_name = f'{file_date_part}_preprocessing_verification.png'
                plt.savefig(os.path.join(viz_path, viz_file_name))
                plt.close()
                print(
                    f"  Verification plot saved for {file} to: {subject_id}/comparative_visualization/{viz_file_name}")


if __name__ == "__main__":
    preprocess_and_visualize()
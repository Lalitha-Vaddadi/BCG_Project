import os
import pandas as pd

base_path = 'C:/Users/lalit/Downloads/dataset/data'
subjects_to_use = {'01', '02', '03'}

def display_raw_bcg():
    for subject_id in sorted(os.listdir(base_path)):
        if subject_id not in subjects_to_use:
            continue

        bcg_path = os.path.join(base_path, subject_id, 'BCG')
        if not os.path.isdir(bcg_path):
            continue

        print(f"\n👤 Subject {subject_id}")
        for file in sorted(os.listdir(bcg_path)):
            if file.endswith('.csv'):
                file_path = os.path.join(bcg_path, file)
                df = pd.read_csv(file_path)
                print(f"\n📄 {file} — First 5 Rows")
                print(df.head())

if __name__ == "__main__":
    display_raw_bcg()

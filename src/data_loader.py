import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

def load_luad_top_mrna(
    feature_repo_path="Main_Dataset/Clustering_datasets/LUAD/Top/LUAD_mRNA.csv",
    label_repo_path=None,
    data_dir="../data"
):
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download mRNA matrix from Hugging Face (cached locally after first run)
    feature_file = hf_hub_download(
        repo_id="AIBIC/MLOmics",
        filename=feature_repo_path,
        repo_type="dataset",
        local_dir=data_dir
    )

    df = pd.read_csv(feature_file, index_col=0)

    # 2. Ensure samples/patients are rows (same check as your scripts)
    if df.shape[0] > df.shape[1]:
        print(f"Detected (Features: {df.shape[0]}, Samples: {df.shape[1]}). Transposing...")
        df = df.T
    else:
        print(f"Detected (Samples: {df.shape[0]}, Features: {df.shape[1]}).")

    # 3. Keep numeric columns and impute any missing values with column medians
    X_df = df.select_dtypes(include=[np.number])
    if X_df.isnull().any().any():
        X_df = X_df.fillna(X_df.median())

    # 4. Optionally load ground-truth labels or clinical survival data if provided
    y = None
    if label_repo_path:
        label_file = hf_hub_download(
            repo_id="AIBIC/MLOmics",
            filename=label_repo_path,
            repo_type="dataset",
            local_dir=data_dir
        )
        labels_df = pd.read_csv(label_file)
        if "Label" in labels_df.columns:
            y = labels_df["Label"].values
        elif "event_observed" in labels_df.columns:
            y = labels_df["event_observed"].values

    return X_df, y
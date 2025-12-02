
"""
Merge KT1 interaction files into a single CSV.
Usage:
    python merge_interactions.py
"""

import os
import pandas as pd
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================
INPUT_DIR = r"C:\Users\spoli\OneDrive\Desktop\KT1"
OUTPUT_FILE = "merged_interactions.csv"

COLUMNS = [
    "timestamp",
    "solving_id",
    "question_id",
    "user_answer",
    "elapsed_time"
]


# ==========================
# FUNCTIONS
# ==========================

def load_one_file(fpath):
    """
    Load one KT1/KT2 CSV file safely and extract user_id from filename.
    Returns a cleaned dataframe or None if error.
    """
    try:
        df = pd.read_csv(fpath)

        # Filter only expected columns that exist in file
        keep_cols = [col for col in COLUMNS if col in df.columns]
        df = df[keep_cols]

        # Extract user ID from filename "u12345.csv"
        fname = os.path.basename(fpath)
        user_id = int(fname.replace("u", "").replace(".csv", ""))

        df["user_id"] = user_id
        return df

    except Exception as e:
        print(f"[ERROR] Failed to load {fpath}: {e}")
        return None


# ==========================
# MAIN MERGE SCRIPT
# ==========================

def main():
    print("Scanning directory:", INPUT_DIR)

    # Gather all CSV files
    files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".csv")
    ]

    print(f"Found {len(files)} user files.\n")

    # Write header once
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(",".join(COLUMNS + ["user_id"]) + "\n")

    # Process files sequentially
    for fpath in tqdm(files, desc="Merging files"):
        df = load_one_file(fpath)
        if df is not None:
            df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)

    print("\n✔ Merge complete!")
    print("Saved to:", OUTPUT_FILE)


# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    main()
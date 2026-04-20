import os
import numpy as np
import pandas as pd

FILES = {
    "llm": "./llm_data/syn_full.csv",
    "gan": "./gan_data/syn_full.csv",
}

ALL_COLS = ["W1", "W2", "W3", "W4", "W5", "W6", "A", "Y"]
BIN_COLS = ["W1", "W2", "W3", "A", "Y"]
LARGE_THRESHOLD = 1e6

def clean_one(label, path):
    print("\n" + "=" * 100)
    print(f"{label.upper()} | reading {path}")

    if not os.path.exists(path):
        print("File not found.")
        return

    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    missing_cols = [c for c in ALL_COLS if c not in df.columns]
    if missing_cols:
        print("Missing columns:", missing_cols)
        return

    num = df[ALL_COLS].copy()
    for c in ALL_COLS:
        num[c] = pd.to_numeric(num[c], errors="coerce")

    bad_mask = pd.Series(False, index=df.index)

    # NaN / non-numeric / inf / huge values
    for c in ALL_COLS:
        bad_mask |= num[c].isna()
        bad_mask |= np.isinf(num[c])
        bad_mask |= ((num[c].abs() > LARGE_THRESHOLD) & (~np.isinf(num[c])))

    # Invalid binary columns
    for c in BIN_COLS:
        bad_mask |= ~num[c].round().isin([0, 1])

    bad_idx = df.index[bad_mask].tolist()
    bad_rows = df.loc[bad_mask].copy()

    print(f"Bad row count: {len(bad_rows)}")
    if len(bad_rows) > 0:
        print("Bad row numbers:")
        print(bad_idx)

        print("\nBad rows:")
        print(bad_rows.to_string())

    clean = df.loc[~bad_mask].copy()

    # force numeric cleanup on saved clean file
    for c in ALL_COLS:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    # round binary columns
    for c in BIN_COLS:
        clean[c] = clean[c].round().astype(int)

    out_path = os.path.join(os.path.dirname(path), "syn_clean.csv")
    clean.to_csv(out_path, index=False)

    print(f"\nClean shape: {clean.shape}")
    print(f"Saved cleaned file to: {out_path}")


if __name__ == "__main__":
    for label, path in FILES.items():
        clean_one(label, path)
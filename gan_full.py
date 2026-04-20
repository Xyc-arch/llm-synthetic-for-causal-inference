import os
import pandas as pd
from ctgan import CTGAN

# Load seed data
data = pd.read_csv("data_seed.csv")

# Keep expected columns only
cols = ["W1", "W2", "W3", "W4", "W5", "W6", "A", "Y"]
data = data[cols].copy()

# Binary / discrete columns in your DGP
discrete_columns = ["W1", "W2", "W3", "A", "Y"]

# Make sure binary columns are integers
for c in discrete_columns:
    data[c] = data[c].round().astype(int)

# Output directory
out_dir = "./gan_data"
os.makedirs(out_dir, exist_ok=True)

# Train CTGAN
ctgan = CTGAN(
    epochs=50,
    verbose=True,
)

ctgan.fit(data, discrete_columns=discrete_columns)

# Sample synthetic data
sample_size = 50000
synthetic_data = ctgan.sample(sample_size)

# Post-process binary columns
for c in discrete_columns:
    synthetic_data[c] = synthetic_data[c].round().astype(int)

# Save
output_path = os.path.join(out_dir, "syn_full.csv")
synthetic_data.to_csv(output_path, index=False)

print(f"Finished. Saved to {output_path}")
print(synthetic_data.head())
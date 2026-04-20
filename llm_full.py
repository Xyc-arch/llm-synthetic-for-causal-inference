import os
import pandas as pd
from be_great import GReaT


def main():
    # Paths
    input_csv = "data_seed.csv"
    output_dir = "./llm_data"
    output_csv = os.path.join(output_dir, "syn_full.csv")
    model_dir = os.path.join(output_dir, "great_checkpoint")

    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    data = pd.read_csv(input_csv)

    # Keep only the columns used in your paper/code
    cols = ["W1", "W2", "W3", "W4", "W5", "W6", "A", "Y"]
    data = data[cols].copy()

    # Optional: cast binary columns explicitly
    for c in ["W1", "W2", "W3", "A", "Y"]:
        data[c] = data[c].astype(int)

    print("Training rows:", len(data))
    print("Columns:", list(data.columns))

    # GReaT model
    # For closer reproduction of older work, use "gpt2" or "distilgpt2".
    # If GPU memory is tight, try distilgpt2 first.
    model = GReaT(
        llm="gpt2",
        batch_size=32,
        epochs=50,
        fp16=True,
        dataloader_num_workers=4,
    )

    # Train
    model.fit(data)

    # Save checkpoint
    model.save(model_dir)
    print(f"Saved model to: {model_dir}")

    # Sample synthetic data
    synthetic_data = model.sample(
        n_samples=50000,
        random_feature_order=True,
        temperature=0.7,
        max_length=1024,
    )

    # Keep only expected columns if extras appear
    synthetic_data = synthetic_data[cols].copy()

    # Round binary-like columns
    for c in ["W1", "W2", "W3", "A", "Y"]:
        synthetic_data[c] = synthetic_data[c].round().astype(int)

    # Save
    synthetic_data.to_csv(output_csv, index=False)
    print(f"Saved synthetic data to: {output_csv}")
    print(synthetic_data.head())


if __name__ == "__main__":
    main()
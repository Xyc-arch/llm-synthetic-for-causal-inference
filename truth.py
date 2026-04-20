import json
from data_generate import generate_dataset

data_truth, ate_true, y1_truth, y0_truth = generate_dataset(
    50000,
    rct=True,
    truth=True
)

truth = {
    "seed": 42,
    "n": 100000,
    "rct": True,
    "ate_true": float(ate_true),
    "y1_truth": float(y1_truth),
    "y0_truth": float(y0_truth),
}

with open("truth.json", "w") as f:
    json.dump(truth, f, indent=4)

print("Saved truth.json")
print(truth)
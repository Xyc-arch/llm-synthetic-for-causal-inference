#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Generating varied simulation datasets..."
python data_vary.py --output-root simulator_vary_data

echo
echo "Generated settings:"
find simulator_vary_data -maxdepth 2 -name "truth.json" -print

echo
echo "Truth summaries:"
for f in simulator_vary_data/*/truth.json; do
    echo "------------------------------------------------------------"
    echo "$f"
    python - <<PY
import json
path = "$f"
with open(path) as fh:
    t = json.load(fh)

print("setting       :", path)
print("d             :", t["d"])
print("n_seed        :", t["n_seed"])
print("n_test        :", t["n_test"])
print("overlap       :", t["overlap"])
print("outcome_mode  :", t["outcome_mode"])
print("seed_data_rct :", t["seed_data_rct"])
print("ate_true      :", round(t["ate_true"], 6))
print("E[Y(1)]       :", round(t["y1_truth"], 6))
print("E[Y(0)]       :", round(t["y0_truth"], 6))
print("pA q05/med/q95:",
      round(t["truth_diagnostics"]["pA_q05"], 4),
      round(t["truth_diagnostics"]["pA_median"], 4),
      round(t["truth_diagnostics"]["pA_q95"], 4))
PY
done

echo
echo "Done."
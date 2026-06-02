#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/syn_causal/more_simulation

ROOT="simulator_vary_data"

# Use "randomized" for causal-purpose synthetic assignment,
# or "estimated" to mimic the learned observational treatment mechanism.
TREATMENT_MODE="randomized"

echo "Running hybrid generation for all settings in: $ROOT"
echo "Treatment mode: $TREATMENT_MODE"
echo

python hybrid_vary.py \
  --root "$ROOT" \
  --treatment-mode "$TREATMENT_MODE" \
  --random-state 42 \
  --skip-if-output-exists

echo
echo "Done."
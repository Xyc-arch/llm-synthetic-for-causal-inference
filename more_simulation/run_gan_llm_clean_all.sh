#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/syn_causal/more_simulation

ROOT="simulator_vary_data"

EPOCHS_GAN=50
EPOCHS_LLM=50

GAN_SAMPLE_SIZE=5000
LLM_SAMPLE_SIZE=1500

LLM="gpt2"
MAX_D_LLM=20

# Original-like GReaT sampling controls.
LLM_SAMPLE_BATCH_SIZE=100
LLM_MAX_LENGTH=1024
LLM_TEMPERATURE=0.7

echo "Running GAN/LLM synthetic generation and cleaning for all settings in: $ROOT"
echo "GAN synthetic sample size: $GAN_SAMPLE_SIZE"
echo "LLM synthetic sample size: $LLM_SAMPLE_SIZE"
echo "LLM max dimension: $MAX_D_LLM"
echo "LLM sample batch size: $LLM_SAMPLE_BATCH_SIZE"
echo "LLM max length: $LLM_MAX_LENGTH"
echo

for DATA_DIR in "$ROOT"/*; do
    if [ ! -d "$DATA_DIR" ]; then
        continue
    fi

    if [ ! -f "$DATA_DIR/data_seed.csv" ]; then
        echo "Skipping $DATA_DIR because data_seed.csv is missing."
        continue
    fi

    GAN_FULL="$DATA_DIR/gan_data/syn_full.csv"
    GAN_CLEAN="$DATA_DIR/gan_data/syn_clean.csv"

    LLM_FULL="$DATA_DIR/llm_data/syn_full.csv"
    LLM_CLEAN="$DATA_DIR/llm_data/syn_clean.csv"
    LLM_CKPT="$DATA_DIR/llm_data/great_checkpoint"

    echo
    echo "================================================================================"
    echo "SETTING: $DATA_DIR"
    echo "================================================================================"

    echo
    echo "[1/3] CTGAN"

    if [ -s "$GAN_FULL" ]; then
        echo "GAN sampled file already exists, skipping CTGAN: $GAN_FULL"
    else
        python gan_full_vary.py \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS_GAN" \
            --sample-size "$GAN_SAMPLE_SIZE"
    fi

    echo
    echo "[2/3] GReaT"

    if [ -s "$LLM_FULL" ]; then
        echo "LLM sampled file already exists, skipping GReaT: $LLM_FULL"
    elif [ -d "$LLM_CKPT" ]; then
        echo "Found existing GReaT checkpoint; sampling from checkpoint."
        python llm_full_vary.py \
            --data-dir "$DATA_DIR" \
            --llm "$LLM" \
            --epochs "$EPOCHS_LLM" \
            --sample-size "$LLM_SAMPLE_SIZE" \
            --sample-batch-size "$LLM_SAMPLE_BATCH_SIZE" \
            --max-length "$LLM_MAX_LENGTH" \
            --temperature "$LLM_TEMPERATURE" \
            --max-d "$MAX_D_LLM" \
            --use-existing-checkpoint
    else
        python llm_full_vary.py \
            --data-dir "$DATA_DIR" \
            --llm "$LLM" \
            --epochs "$EPOCHS_LLM" \
            --sample-size "$LLM_SAMPLE_SIZE" \
            --sample-batch-size "$LLM_SAMPLE_BATCH_SIZE" \
            --max-length "$LLM_MAX_LENGTH" \
            --temperature "$LLM_TEMPERATURE" \
            --max-d "$MAX_D_LLM"
    fi

    echo
    echo "[3/3] Cleaning synthetic outputs"

    if [ -s "$GAN_CLEAN" ] && { [ -s "$LLM_CLEAN" ] || [ ! -s "$LLM_FULL" ]; }; then
        echo "Cleaned files already exist where available, skipping cleaning."
        echo "GAN clean: $GAN_CLEAN"
        echo "LLM clean: $LLM_CLEAN"
    else
        python clean_syn_vary.py \
            --data-dir "$DATA_DIR"
    fi

    echo
    echo "Finished setting: $DATA_DIR"
done

echo
echo "All done."
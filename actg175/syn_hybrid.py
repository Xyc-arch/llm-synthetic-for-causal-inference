#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# This script lives in /home/ubuntu/syn_causal/actg175/
ACTG_DIR = Path(__file__).resolve().parent

# Real ACTG file
RAW_FILE = ACTG_DIR / "actg175.csv"

# Synthetic folders
LLM_DIR = ACTG_DIR / "llm_data"
CTGAN_DIR = ACTG_DIR / "ctgan_data"

# Read filtered synthetic covariates
LLM_INPUT = LLM_DIR / "syn_filter.csv"
CTGAN_INPUT = CTGAN_DIR / "syn_filter.csv"

# Save hybrid outputs
LLM_OUTPUT = LLM_DIR / "syn_hybrid.csv"
CTGAN_OUTPUT = CTGAN_DIR / "syn_hybrid.csv"

W_VARS = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
CAT_VARS = [
    "hemo", "homo", "drugs", "oprior", "z30", "zprior",
    "race", "gender", "str2", "strat", "symptom"
]

A_VAR = "A"
Y_VAR = "cd420"


def load_and_prepare_seed() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing ACTG file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    if "arms" not in df.columns:
        raise ValueError("Column 'arms' not found in ACTG175 data.")

    # Same coding as your R code:
    # keep only arms 1 and 2
    df = df[df["arms"].isin([1, 2])].copy()

    # A = 0 if arms==1, A = 1 if arms==2
    df[A_VAR] = (df["arms"] == 2).astype(int)

    needed = W_VARS + [A_VAR, Y_VAR]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw ACTG file: {missing}")

    df = df[needed].dropna().copy()

    # Cast continuous columns to float before scaling
    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Standardize continuous vars like your R code
    scaler = StandardScaler()
    df[CONT_VARS] = scaler.fit_transform(df[CONT_VARS])

    # Cast discrete columns
    for c in CAT_VARS + [A_VAR]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)

    return df


def build_propensity_model():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS),
            ("num", "passthrough", CONT_VARS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                min_samples_leaf=5,
                n_jobs=-1
            )),
        ]
    )


def build_outcome_model():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS + [A_VAR]),
            ("num", "passthrough", CONT_VARS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("reg", RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                min_samples_leaf=5,
                n_jobs=-1
            )),
        ]
    )


def fit_models(seed_df: pd.DataFrame):
    g_model = build_propensity_model()
    g_model.fit(seed_df[W_VARS], seed_df[A_VAR])

    q_model = build_outcome_model()
    q_model.fit(seed_df[W_VARS + [A_VAR]], seed_df[Y_VAR])

    return g_model, q_model


def project_categorical_support(seed_df: pd.DataFrame, syn_w: pd.DataFrame) -> pd.DataFrame:
    syn_w = syn_w.copy()

    for c in CAT_VARS:
        syn_w[c] = pd.to_numeric(syn_w[c], errors="coerce").round().astype(int)
        observed = sorted(seed_df[c].dropna().unique().tolist())
        syn_w[c] = syn_w[c].apply(lambda x: min(observed, key=lambda z: abs(z - x)))

    return syn_w


def coerce_continuous(syn_w: pd.DataFrame) -> pd.DataFrame:
    syn_w = syn_w.copy()
    for c in CONT_VARS:
        syn_w[c] = pd.to_numeric(syn_w[c], errors="coerce").astype(float)
    return syn_w


def make_hybrid(seed_df: pd.DataFrame, syn_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    missing = [c for c in W_VARS if c not in syn_df.columns]
    if missing:
        raise ValueError(f"Synthetic file missing W columns: {missing}")

    syn_w = syn_df[W_VARS].copy()
    syn_w = coerce_continuous(syn_w)
    syn_w = project_categorical_support(seed_df, syn_w)

    # Drop any rows that somehow still contain missing values
    syn_w = syn_w.dropna().reset_index(drop=True)

    g_model, q_model = fit_models(seed_df)

    # Sample A ~ g(A|W)
    pA = g_model.predict_proba(syn_w)[:, 1]
    pA = np.clip(pA, 1e-6, 1 - 1e-6)
    A_syn = rng.binomial(1, pA, size=len(syn_w))

    # Predict Y ~ Q(Y|A,W)
    q_input = syn_w.copy()
    q_input[A_VAR] = A_syn
    y_pred = q_model.predict(q_input)

    # Clip to observed support
    y_min = float(seed_df[Y_VAR].min())
    y_max = float(seed_df[Y_VAR].max())
    y_pred = np.clip(y_pred, y_min, y_max)

    hybrid = syn_w.copy()
    hybrid[A_VAR] = A_syn.astype(int)
    hybrid[Y_VAR] = y_pred

    return hybrid


def process_one(name: str, input_file: Path, output_file: Path, seed_df: pd.DataFrame):
    if not input_file.exists():
        raise FileNotFoundError(f"Missing synthetic input for {name}: {input_file}")

    syn_df = pd.read_csv(input_file)
    rng = np.random.default_rng(42)

    hybrid_df = make_hybrid(seed_df, syn_df, rng)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_csv(output_file, index=False)

    print(f"[{name}] input:  {input_file}")
    print(f"[{name}] output: {output_file}")
    print(f"[{name}] hybrid shape: {hybrid_df.shape}")
    print(hybrid_df.head().to_string(index=False))
    print()


def main():
    seed_df = load_and_prepare_seed()

    print(f"Loaded raw ACTG data from: {RAW_FILE}")
    print(f"Prepared seed shape: {seed_df.shape}")
    print()

    process_one("llm", LLM_INPUT, LLM_OUTPUT, seed_df)
    process_one("ctgan", CTGAN_INPUT, CTGAN_OUTPUT, seed_df)


if __name__ == "__main__":
    main()
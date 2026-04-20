#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_aipw_continuous_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
) -> float:
    data = data.copy()

    # Coerce types
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce").astype(float)
    data[treatment_col] = pd.to_numeric(
        data[treatment_col], errors="coerce"
    ).round().astype(int)

    needed = list(covariates) + [treatment_col, outcome_col]
    data = data[needed].dropna().reset_index(drop=True)

    A = data[treatment_col].to_numpy(dtype=int)
    Y = data[outcome_col].to_numpy(dtype=float)

    # Propensity score model
    rf_ps = RandomForestClassifier(random_state=random_state)
    rf_ps.fit(data[list(covariates)], A)
    ps = np.clip(
        rf_ps.predict_proba(data[list(covariates)])[:, 1],
        clip_min,
        1.0 - clip_min,
    )

    # Outcome regression model for continuous Y
    rf_out = RandomForestRegressor(random_state=random_state)
    X_out = data[list(covariates) + [treatment_col]]
    rf_out.fit(X_out, Y)

    X1 = data[list(covariates)].copy()
    X1[treatment_col] = 1
    m1 = rf_out.predict(X1)

    X0 = data[list(covariates)].copy()
    X0[treatment_col] = 0
    m0 = rf_out.predict(X0)

    est = np.mean(
        m1 - m0
        + A * (Y - m1) / ps
        - (1 - A) * (Y - m0) / (1 - ps)
    )
    return float(est)


def estimate_aipw_continuous(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_aipw_continuous_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        clip_min=clip_min,
    )
    if verbose:
        print(f"{file_path}: AIPW continuous ATE = {est}")
    return est
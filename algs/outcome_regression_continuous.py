#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_outcome_regression_continuous_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
) -> float:
    data = data.copy()

    # Coerce types
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce").astype(float)
    data[treatment_col] = pd.to_numeric(
        data[treatment_col], errors="coerce"
    ).round().astype(int)

    needed = list(covariates) + [treatment_col, outcome_col]
    data = data[needed].dropna().reset_index(drop=True)

    # Outcome regression for continuous Y
    rf_out = RandomForestRegressor(random_state=random_state)
    X_out = data[list(covariates) + [treatment_col]]
    Y = data[outcome_col].to_numpy(dtype=float)
    rf_out.fit(X_out, Y)

    # Predict counterfactual means under A=1 and A=0
    X1 = data[list(covariates)].copy()
    X1[treatment_col] = 1
    m1 = rf_out.predict(X1)

    X0 = data[list(covariates)].copy()
    X0[treatment_col] = 0
    m0 = rf_out.predict(X0)

    return float(np.mean(m1 - m0))


def estimate_outcome_regression_continuous(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_outcome_regression_continuous_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
    )
    if verbose:
        print(f"{file_path}: Outcome-regression continuous ATE = {est}")
    return est
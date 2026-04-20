import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_aipw_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
) -> float:
    data = data.copy()
    data[outcome_col] = data[outcome_col].round().astype(int)
    data[treatment_col] = data[treatment_col].round().astype(int)

    rf_ps = RandomForestClassifier(random_state=random_state)
    rf_ps.fit(data[covariates], data[treatment_col])
    ps = np.clip(
        rf_ps.predict_proba(data[covariates])[:, 1],
        clip_min,
        1 - clip_min,
    )

    rf_out = RandomForestClassifier(random_state=random_state)
    X_out = data[list(covariates) + [treatment_col]]
    rf_out.fit(X_out, data[outcome_col])

    X1 = data[list(covariates)].copy()
    X1[treatment_col] = 1
    m1 = rf_out.predict_proba(X1)[:, 1]

    X0 = data[list(covariates)].copy()
    X0[treatment_col] = 0
    m0 = rf_out.predict_proba(X0)[:, 1]

    A = data[treatment_col].to_numpy()
    Y = data[outcome_col].to_numpy()

    est = np.mean(m1 - m0 + A * (Y - m1) / ps - (1 - A) * (Y - m0) / (1 - ps))
    return float(est)


def estimate_aipw(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_aipw_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        clip_min=clip_min,
    )
    if verbose:
        print(f"{file_path}: AIPW ATE = {est}")
    return est
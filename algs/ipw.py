import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_ipw_df(
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

    A = data[treatment_col].to_numpy()
    Y = data[outcome_col].to_numpy()

    w_treated = A / ps
    w_control = (1 - A) / (1 - ps)

    y1 = np.sum(w_treated * Y) / np.sum(w_treated)
    y0 = np.sum(w_control * Y) / np.sum(w_control)

    return float(y1 - y0)


def estimate_ipw(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_ipw_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        clip_min=clip_min,
    )
    if verbose:
        print(f"{file_path}: IPW ATE = {est}")
    return est
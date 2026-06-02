import pandas as pd
import numpy as np

from algs.lrnr import make_g_learner, predict_binary_prob


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_ipw_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    g_learner: str = "rf",
    outcome_type: str = "binary",
) -> float:
    """
    Stabilized/Hajek IPW estimator for ATE.

    Supports:
      - binary outcome
      - continuous outcome

    Treatment A is assumed binary.
    """
    data = data.copy()
    covariates = list(covariates)

    for c in covariates:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data[treatment_col] = pd.to_numeric(data[treatment_col], errors="coerce")
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce")

    data = data.dropna(subset=covariates + [treatment_col, outcome_col]).copy()

    data[treatment_col] = data[treatment_col].round().astype(int)

    if outcome_type == "binary":
        data[outcome_col] = data[outcome_col].round().astype(int)
    elif outcome_type == "continuous":
        data[outcome_col] = data[outcome_col].astype(float)
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")

    # g(W) = P(A=1 | W)
    g_model = make_g_learner(g_learner, random_state=random_state)
    g_model.fit(data[covariates], data[treatment_col])

    ps = predict_binary_prob(g_model, data[covariates])
    ps = np.clip(ps, clip_min, 1.0 - clip_min)

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
    g_learner: str = "rf",
    outcome_type: str = "binary",
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
        g_learner=g_learner,
        outcome_type=outcome_type,
    )

    if verbose:
        print(f"{file_path}: IPW ATE = {est}")

    return est
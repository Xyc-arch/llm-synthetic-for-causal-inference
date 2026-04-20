import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logit, expit
from sklearn.ensemble import RandomForestClassifier


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_tmle_df(
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

    rf_g = RandomForestClassifier(random_state=random_state)
    rf_g.fit(data[list(covariates)], data[treatment_col])
    g1w = np.clip(rf_g.predict_proba(data[list(covariates)])[:, 1], clip_min, 1 - clip_min)
    g0w = 1.0 - g1w

    rf_q = RandomForestClassifier(random_state=random_state)
    X_q = data[list(covariates) + [treatment_col]]
    rf_q.fit(X_q, data[outcome_col])

    X_obs = data[list(covariates) + [treatment_col]].copy()
    QAW = np.clip(rf_q.predict_proba(X_obs)[:, 1], clip_min, 1 - clip_min)

    X1 = data[list(covariates)].copy()
    X1[treatment_col] = 1
    Q1W = np.clip(rf_q.predict_proba(X1)[:, 1], clip_min, 1 - clip_min)

    X0 = data[list(covariates)].copy()
    X0[treatment_col] = 0
    Q0W = np.clip(rf_q.predict_proba(X0)[:, 1], clip_min, 1 - clip_min)

    A = data[treatment_col].to_numpy()
    Y = data[outcome_col].to_numpy()

    HAW = A / g1w - (1 - A) / g0w
    H1W = 1.0 / g1w
    H0W = -1.0 / g0w

    offset = logit(QAW)
    fluctuation_model = sm.GLM(
        Y,
        HAW.reshape(-1, 1),
        family=sm.families.Binomial(),
        offset=offset,
    )
    fluctuation_result = fluctuation_model.fit()
    epsilon = float(fluctuation_result.params[0])

    Q1W_star = expit(logit(Q1W) + epsilon * H1W)
    Q0W_star = expit(logit(Q0W) + epsilon * H0W)

    tmle_est = np.mean(Q1W_star - Q0W_star)
    return float(tmle_est)


def estimate_tmle(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_tmle_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        clip_min=clip_min,
    )
    if verbose:
        print(f"{file_path}: TMLE ATE = {est}")
    return est
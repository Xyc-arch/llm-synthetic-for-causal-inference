import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logit, expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def _bound(x, clip_min: float = 1e-6):
    x = np.asarray(x, dtype=float)
    return np.clip(x, clip_min, 1.0 - clip_min)


def estimate_tmle_continuous_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
) -> float:
    data = data.copy()

    # treatment should be binary
    data[treatment_col] = data[treatment_col].round().astype(int)

    # continuous outcome
    Y = pd.to_numeric(data[outcome_col], errors="coerce").to_numpy(dtype=float)
    A = data[treatment_col].to_numpy(dtype=int)

    # scale Y to [0,1]
    y_min = float(np.min(Y))
    y_max = float(np.max(Y))
    y_range = y_max - y_min
    if y_range <= 0:
        raise ValueError("Outcome has zero range; cannot run continuous TMLE.")

    Y_unit = (Y - y_min) / y_range
    Y_unit = np.clip(Y_unit, 0.0, 1.0)

    # g model
    rf_g = RandomForestClassifier(random_state=random_state)
    rf_g.fit(data[list(covariates)], data[treatment_col])
    g1w = np.clip(rf_g.predict_proba(data[list(covariates)])[:, 1], clip_min, 1 - clip_min)
    g0w = 1.0 - g1w

    # Q model on unit scale
    data_q = data.copy()
    data_q["_Y_unit"] = Y_unit

    rf_q = RandomForestRegressor(random_state=random_state)
    X_q = data_q[list(covariates) + [treatment_col]]
    rf_q.fit(X_q, data_q["_Y_unit"])

    X_obs = data_q[list(covariates) + [treatment_col]].copy()
    QAW = _bound(rf_q.predict(X_obs), clip_min)

    X1 = data_q[list(covariates)].copy()
    X1[treatment_col] = 1
    Q1W = _bound(rf_q.predict(X1), clip_min)

    X0 = data_q[list(covariates)].copy()
    X0[treatment_col] = 0
    Q0W = _bound(rf_q.predict(X0), clip_min)

    # clever covariates
    HAW = A / g1w - (1 - A) / g0w
    H1W = 1.0 / g1w
    H0W = -1.0 / g0w

    # fluctuation on logit scale
    offset = logit(QAW)
    fluctuation_model = sm.GLM(
        Y_unit,
        HAW.reshape(-1, 1),
        family=sm.families.Binomial(),
        offset=offset,
    )
    fluctuation_result = fluctuation_model.fit()
    epsilon = float(fluctuation_result.params[0])

    Q1W_star_unit = expit(logit(Q1W) + epsilon * H1W)
    Q0W_star_unit = expit(logit(Q0W) + epsilon * H0W)

    # transform back to original outcome scale
    Q1W_star = Q1W_star_unit * y_range + y_min
    Q0W_star = Q0W_star_unit * y_range + y_min

    tmle_est = np.mean(Q1W_star - Q0W_star)
    return float(tmle_est)


def estimate_tmle_continuous(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_tmle_continuous_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        clip_min=clip_min,
    )
    if verbose:
        print(f"{file_path}: TMLE continuous ATE = {est}")
    return est
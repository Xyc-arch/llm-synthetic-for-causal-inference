import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.special import logit, expit

from algs.lrnr import make_g_learner, make_q_learner, predict_binary_prob, predict_q


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_tmle_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    clip_min: float = 1e-6,
    g_learner: str = "rf",
    q_learner: str = "rf",
    outcome_type: str = "binary",
) -> float:
    """
    Binary-outcome TMLE estimator for ATE.

    Current implementation uses logistic fluctuation:

        logit(Q*) = logit(Q) + epsilon * H

    so outcome_type must be "binary".

    Treatment A is assumed binary.
    """
    if outcome_type != "binary":
        raise NotImplementedError(
            "Current TMLE implementation uses logistic fluctuation and only supports binary outcomes."
        )

    data = data.copy()
    covariates = list(covariates)

    for c in covariates:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data[treatment_col] = pd.to_numeric(data[treatment_col], errors="coerce")
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce")

    data = data.dropna(subset=covariates + [treatment_col, outcome_col]).copy()

    data[treatment_col] = data[treatment_col].round().astype(int)
    data[outcome_col] = data[outcome_col].round().astype(int)

    # g(W) = P(A=1 | W)
    g_model = make_g_learner(g_learner, random_state=random_state)
    g_model.fit(data[covariates], data[treatment_col])

    g1w = predict_binary_prob(g_model, data[covariates])
    g1w = np.clip(g1w, clip_min, 1.0 - clip_min)
    g0w = 1.0 - g1w

    # Q(A,W) = P(Y=1 | A,W)
    q_model = make_q_learner(
        q_learner,
        outcome_type=outcome_type,
        random_state=random_state,
    )

    X_q = data[covariates + [treatment_col]]
    q_model.fit(X_q, data[outcome_col])

    X_obs = data[covariates + [treatment_col]].copy()
    QAW = predict_q(q_model, X_obs, outcome_type=outcome_type)
    QAW = np.clip(QAW, clip_min, 1.0 - clip_min)

    X1 = data[covariates].copy()
    X1[treatment_col] = 1
    Q1W = predict_q(q_model, X1, outcome_type=outcome_type)
    Q1W = np.clip(Q1W, clip_min, 1.0 - clip_min)

    X0 = data[covariates].copy()
    X0[treatment_col] = 0
    Q0W = predict_q(q_model, X0, outcome_type=outcome_type)
    Q0W = np.clip(Q0W, clip_min, 1.0 - clip_min)

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
    g_learner: str = "rf",
    q_learner: str = "rf",
    outcome_type: str = "binary",
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
        g_learner=g_learner,
        q_learner=q_learner,
        outcome_type=outcome_type,
    )

    if verbose:
        print(f"{file_path}: TMLE ATE = {est}")

    return est
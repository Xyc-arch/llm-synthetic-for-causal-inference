import pandas as pd

from algs.lrnr import make_q_learner, predict_q


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_outcome_regression_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    q_learner: str = "rf",
    outcome_type: str = "binary",
) -> float:
    """
    Outcome regression / plug-in estimator for ATE.

    Supports:
      - binary outcome: Q estimates P(Y=1 | A,W)
      - continuous outcome: Q estimates E[Y | A,W)

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

    # Q(A,W) = E[Y | A,W]
    q_model = make_q_learner(
        q_learner,
        outcome_type=outcome_type,
        random_state=random_state,
    )

    X_out = data[covariates + [treatment_col]]
    q_model.fit(X_out, data[outcome_col])

    X1 = data[covariates].copy()
    X1[treatment_col] = 1
    m1 = predict_q(q_model, X1, outcome_type=outcome_type)

    X0 = data[covariates].copy()
    X0[treatment_col] = 0
    m0 = predict_q(q_model, X0, outcome_type=outcome_type)

    return float((m1 - m0).mean())


def estimate_outcome_regression(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    q_learner: str = "rf",
    outcome_type: str = "binary",
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)

    est = estimate_outcome_regression_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
        q_learner=q_learner,
        outcome_type=outcome_type,
    )

    if verbose:
        print(f"{file_path}: Outcome-regression ATE = {est}")

    return est
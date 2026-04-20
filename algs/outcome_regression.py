import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).copy()


def estimate_outcome_regression_df(
    data: pd.DataFrame,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
) -> float:
    data = data.copy()
    data[outcome_col] = data[outcome_col].round().astype(int)
    data[treatment_col] = data[treatment_col].round().astype(int)

    rf_out = RandomForestClassifier(random_state=random_state)
    X_out = data[list(covariates) + [treatment_col]]
    rf_out.fit(X_out, data[outcome_col])

    X1 = data[list(covariates)].copy()
    X1[treatment_col] = 1
    m1 = rf_out.predict_proba(X1)[:, 1]

    X0 = data[list(covariates)].copy()
    X0[treatment_col] = 0
    m0 = rf_out.predict_proba(X0)[:, 1]

    return float((m1 - m0).mean())


def estimate_outcome_regression(
    file_path: str,
    covariates,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    random_state: int = 42,
    verbose: bool = True,
) -> float:
    data = _load_data(file_path)
    est = estimate_outcome_regression_df(
        data=data,
        covariates=covariates,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        random_state=random_state,
    )
    if verbose:
        print(f"{file_path}: Outcome-regression ATE = {est}")
    return est
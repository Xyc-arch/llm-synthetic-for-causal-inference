import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def make_g_learner(kind: str = "rf", random_state: int = 42):
    """
    Learner for g(W) = P(A=1 | W).

    A is assumed binary, so this should usually be a classifier.
    """
    if kind == "rf":
        return RandomForestClassifier(random_state=random_state)

    if kind in ["logistic", "logit", "lr"]:
        return LogisticRegression(max_iter=1000)

    raise ValueError(f"Unknown g learner kind: {kind}")


def make_q_learner(
    kind: str = "rf",
    outcome_type: str = "binary",
    random_state: int = 42,
):
    """
    Learner for Q(A,W) = E[Y | A,W].

    outcome_type:
      - "binary": classifier, returns P(Y=1 | A,W)
      - "continuous": regressor, returns E[Y | A,W)
    """
    if outcome_type == "binary":
        if kind == "rf":
            return RandomForestClassifier(random_state=random_state)

        if kind in ["logistic", "logit", "lr"]:
            return LogisticRegression(max_iter=1000)

    if outcome_type == "continuous":
        if kind == "rf":
            return RandomForestRegressor(random_state=random_state)

        if kind in ["linear", "lm", "lr"]:
            return LinearRegression()

    raise ValueError(f"Unknown Q learner kind={kind}, outcome_type={outcome_type}")


def predict_binary_prob(model, X):
    """
    Return P(Y=1 | X) or P(A=1 | X) from a classifier.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # Normal binary classifier case
        if proba.shape[1] == 2:
            return proba[:, 1]

        # Degenerate case: training sample contains only one class
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1:
            return np.ones(len(X)) * float(classes[0])

    # Fallback for models without predict_proba
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def predict_q(model, X, outcome_type: str = "binary"):
    """
    Return Q predictions.

    For binary Y, returns predicted probability of Y=1.
    For continuous Y, returns predicted mean.
    """
    if outcome_type == "binary":
        return predict_binary_prob(model, X)

    if outcome_type == "continuous":
        return np.asarray(model.predict(X), dtype=float)

    raise ValueError(f"Unknown outcome_type: {outcome_type}")
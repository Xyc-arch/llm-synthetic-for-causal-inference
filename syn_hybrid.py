import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def hybrid(syn_file, data_seed_path, data_path):
    covs = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']

    seed = pd.read_csv(data_seed_path)
    syn = pd.read_csv(os.path.join(data_path, syn_file)).drop(columns=['A', 'Y'], errors='ignore')

    # Keep only needed columns
    seed = seed[covs + ['A', 'Y']].copy()
    syn = syn[covs].copy()

    # Train PS model using RF
    X_ps = seed[covs]
    ps_model = RandomForestClassifier(random_state=42)
    ps_model.fit(X_ps, seed['A'])

    # Predict propensity scores and assign A
    syn['ps'] = np.clip(
        ps_model.predict_proba(syn[covs])[:, 1],
        1e-6,
        1 - 1e-6
    )
    syn['A'] = syn['ps'].apply(lambda p: np.random.binomial(1, p))

    # Train outcome model using RF
    X_outcome = seed[covs + ['A']]
    outcome_model = RandomForestClassifier(random_state=42)
    outcome_model.fit(X_outcome, seed['Y'])

    # Predict outcome probabilities and assign Y
    X_syn_outcome = syn[covs + ['A']]
    syn['y_prob'] = outcome_model.predict_proba(X_syn_outcome)[:, 1]
    syn['Y'] = syn['y_prob'].apply(lambda p: np.random.binomial(1, p))

    syn_hybrid = syn[covs + ['A', 'Y']]
    out_path = os.path.join(data_path, "syn_hybrid.csv")
    syn_hybrid.to_csv(out_path, index=False)
    print(f"Synthetic hybrid data saved as {out_path}")

    return syn_hybrid


if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}

    for i in [0, 1]:
        gen = gens[i]
        parent_path = f"./{gen}_data"
        print(f"\nRunning hybrid generation for: {gen}")
        hybrid("syn_clean.csv", "data_seed.csv", parent_path)
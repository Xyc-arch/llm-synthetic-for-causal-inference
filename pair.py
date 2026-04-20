import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def augment_data(orig_dat, syn_dat, data_path):
    # Load original and synthetic data
    orig = pd.read_csv(orig_dat)
    syn = pd.read_csv(data_path + syn_dat)
    
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(orig[covariates], orig['A'])
    orig['ps'] = rf.predict_proba(orig[covariates])[:, 1]
    
    thres = 0.001
    extreme_low = orig[orig['ps'] < thres]
    extreme_high = orig[orig['ps'] > (1 - thres)]
    print("Extreme low (ps < {:.3f}): {}".format(thres, len(extreme_low)))
    print("Extreme high (ps > {:.3f}): {}".format(1 - thres, len(extreme_high)))
    
    # Standardize covariates using orig's mean and std
    means = orig[covariates].mean()
    stds = orig[covariates].std()
    orig_std = orig.copy()
    syn_std = syn.copy()
    orig_std[covariates] = (orig[covariates] - means) / stds
    syn_std[covariates] = (syn[covariates] - means) / stds
    
    matched_candidates = []
    used_idx = set()
    
    # For extreme low samples, look for synthetic samples with A==1
    for idx, row in orig_std[orig_std['ps'] < thres].iterrows():
        candidates = syn_std[(syn_std['A'] == 1) & (~syn_std.index.isin(used_idx))]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[covariates] - row[covariates])**2).sum(axis=1))
        best_idx = distances.idxmin()
        matched_candidates.append(syn.loc[best_idx])
        used_idx.add(best_idx)
    
    # For extreme high samples, look for synthetic samples with A==0
    for idx, row in orig_std[orig_std['ps'] > (1 - thres)].iterrows():
        candidates = syn_std[(syn_std['A'] == 0) & (~syn_std.index.isin(used_idx))]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[covariates] - row[covariates])**2).sum(axis=1))
        best_idx = distances.idxmin()
        matched_candidates.append(syn.loc[best_idx])
        used_idx.add(best_idx)
    
    # Augment original data with the matched synthetic samples
    if matched_candidates:
        matched_df = pd.DataFrame(matched_candidates)
        augmented = pd.concat([orig, matched_df], ignore_index=True)
    else:
        augmented = orig.copy()
    
    print("Total samples in augmented data:", len(augmented))
    augmented.to_csv(data_path + "pair.csv", index=False)
    return augmented

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    for i in range(len(gens)):
        gen = gens[i]
        parent_path = "./{}_data/".format(gen)
        augment_data("data.csv", "syn_hybrid.csv", parent_path)

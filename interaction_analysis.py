
from typing import Sequence
import numpy as np
import pandas as pd

def _shap_vals_array(shap_values):
    try:
        vals = getattr(shap_values, 'values', shap_values)
    except Exception:
        vals = shap_values
    arr = np.array(vals)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    return arr

def compute_shap_interaction_matrix(explainer, shap_values, feature_names: Sequence[str]):

    try:
        # shap interaction values often available via explainer.shap_interaction_values(X)
        inter_vals = explainer.shap_interaction_values  # might be callable or precomputed
        if callable(inter_vals):
            inter = explainer.shap_interaction_values  # function

            raise RuntimeError('Explainer requires X to compute interactions; use compute_shap_interaction_matrix_with_X instead')
        else:
            inter = inter_vals
        arr = np.array(inter)
        if arr.ndim == 4:
            # multiclass: average over classes -> (n_samples, n_features, n_features)
            arr = arr.mean(axis=1)
        mean_abs = np.mean(np.abs(arr), axis=0)
        df = pd.DataFrame(mean_abs, index=feature_names, columns=feature_names)
        return df
    except Exception:
        raise RuntimeError('True SHAP interaction values unavailable from explainer; use interaction_proxy_by_shapcorr instead')

def interaction_proxy_by_shapcorr(shap_values, feature_names: Sequence[str]):
    arr = _shap_vals_array(shap_values)  # (n_samples, n_features)
    # compute correlation matrix
    with np.errstate(invalid='ignore'):
        corr = np.corrcoef(arr, rowvar=False)
    # take absolute value and replace nan with 0
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    df = pd.DataFrame(corr, index=feature_names, columns=feature_names)
    return df


from typing import Sequence, List
import numpy as np
import pandas as pd

def _extract_shap_array(shap_values):

    try:
        vals = getattr(shap_values, 'values', shap_values)
    except Exception:
        vals = shap_values
    arr = np.array(vals)
    if arr.ndim == 3:
        # multiclass: sum/mean across classes (use mean absolute)
        # shape (n_samples, n_classes, n_features) -> take mean across classes
        arr = arr.mean(axis=1)
    return arr  # shape (n_samples, n_features)

def global_shap_importance(shap_values, feature_names: Sequence[str]):
    """Return DataFrame with feature, mean_abs_shap, std_abs_shap sorted desc."""
    arr = _extract_shap_array(shap_values)
    mean_abs = np.mean(np.abs(arr), axis=0)
    std_abs = np.std(np.abs(arr), axis=0)
    df = pd.DataFrame({
        'feature': list(feature_names),
        'mean_abs_shap': mean_abs,
        'std_abs_shap': std_abs
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return df

def top_k_features(df, k=10):
    return df.head(k)


from typing import Optional, Sequence, List, Dict, Any
import numpy as np
import pandas as pd

def _get_values(shap_values):
    try:
        vals = getattr(shap_values, 'values', shap_values)
    except Exception:
        vals = shap_values
    return np.array(vals)

def explain_instance_shap(shap_values, X: pd.DataFrame, idx: int = 0, feature_names: Optional[Sequence[str]] = None):
    """Return per-feature contribution for a single instance as sorted DataFrame."""
    arr = _get_values(shap_values)
    if arr.ndim == 3:
        # multiclass -> average across classes for local explanation
        arr = arr.mean(axis=1)
    instance_vals = arr[idx]
    if feature_names is None:
        feature_names = list(X.columns)
    df = pd.DataFrame({
        'feature': list(feature_names),
        'shap_value': instance_vals,
        'abs_shap': np.abs(instance_vals)
    }).sort_values('abs_shap', ascending=False).reset_index(drop=True)
    return df

def local_summary(shap_values, X: pd.DataFrame, idx_list: List[int], feature_names: Optional[Sequence[str]] = None):
    rows = []
    for i in idx_list:
        df_i = explain_instance_shap(shap_values, X, idx=i, feature_names=feature_names)
        top = df_i.head(10).to_dict(orient='records')
        rows.append({'idx': int(i), 'top_contributors': top})
    return pd.DataFrame(rows)

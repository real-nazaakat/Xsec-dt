
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from sklearn.inspection import permutation_importance

def compute_shap_or_permutation(model: Any, X: pd.DataFrame, feature_names: Optional[Sequence[str]] = None,
                                random_state: int = 42, nsamples: int = 100) -> Dict:

    if feature_names is None:
        feature_names = list(X.columns)

    # Try SHAP
    if HAS_SHAP:
        try:
            # prefer tree explainer for tree models for speed/accuracy
            try:
                explainer = shap.Explainer(model, X, feature_names=feature_names)
            except Exception:
                # fallback: explicit tree/linear explainer selection
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(len(X), nsamples)))
            shap_values = explainer(X)
            # shap_values may be an Explanation object; extract values as numpy (handle binary multiclass)
            try:
                vals = shap_values.values  # shape (n_samples, n_features) or (n_samples, n_classes, n_features)
            except Exception:
                vals = np.array(shap_values)
            return {
                'method': 'shap',
                'explainer': explainer,
                'shap_values': vals,
                'permutation_importance': None,
                'feature_names': list(feature_names)
            }
        except Exception:
            # fall through to permutation below
            pass

    # Permutation importance fallback
    try:
        # Use predict_proba for probabilistic models if available
        if hasattr(model, 'predict_proba'):
            scorer = None  # let sklearn choose default (accuracy) or user can change externally
            r = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=random_state, n_jobs=1)
        else:
            r = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=random_state, n_jobs=1)
        perm_df = pd.DataFrame({
            'feature': list(feature_names),
            'importance_mean': r.importances_mean,
            'importance_std': r.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
        return {
            'method': 'permutation',
            'explainer': None,
            'shap_values': None,
            'permutation_importance': perm_df,
            'feature_names': list(feature_names)
        }
    except Exception as e:
        raise RuntimeError(f'Failed to compute SHAP or permutation importance: {e}')

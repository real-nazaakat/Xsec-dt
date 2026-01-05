
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


def impute_missing(df: pd.DataFrame, numeric_columns: List[str], strategy: str = 'median',
                   fill_value: Optional[float] = None) -> pd.DataFrame:

    d = df.copy()
    for col in numeric_columns:
        if col not in d.columns:
            d[col] = np.nan
        if strategy == 'mean':
            val = d[col].mean(skipna=True)
            if pd.isna(val):
                val = 0.0
        elif strategy == 'median':
            val = d[col].median(skipna=True)
            if pd.isna(val):
                val = 0.0
        elif strategy == 'constant':
            val = float(fill_value) if fill_value is not None else 0.0
        else:
            raise ValueError('Unknown strategy: ' + str(strategy))
        d[col] = d[col].fillna(val)
    return d


def min_max_scale(df: pd.DataFrame, numeric_columns: List[str], clip: bool = True) -> pd.DataFrame:

    d = df.copy()
    for col in numeric_columns:
        if col not in d.columns:
            d[col] = 0.0
        arr = d[col].astype(float)
        mn = arr.min(skipna=True)
        mx = arr.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            scaled = np.zeros_like(arr, dtype=float)
        else:
            scaled = (arr - mn) / (mx - mn)
            if clip:
                scaled = np.clip(scaled, 0.0, 1.0)
        d[col] = scaled
    return d


def zscore_scale(df: pd.DataFrame, numeric_columns: List[str], clip_std: Optional[float] = None) -> pd.DataFrame:
    d = df.copy()
    for col in numeric_columns:
        if col not in d.columns:
            d[col] = 0.0
        arr = d[col].astype(float)
        mu = arr.mean(skipna=True)
        sigma = arr.std(skipna=True)
        if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            zs = np.zeros_like(arr, dtype=float)
        else:
            zs = (arr - mu) / sigma
            if clip_std is not None:
                zs = np.clip(zs, -clip_std, clip_std)
        d[col] = zs
    return d

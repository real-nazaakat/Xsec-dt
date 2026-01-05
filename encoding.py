
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


def one_hot_encode(df: pd.DataFrame, categorical_columns: List[str], drop_first: bool = False,
                   treat_missing_as_category: bool = True) -> Tuple[pd.DataFrame, List[str]]:

    d = df.copy()
    for col in categorical_columns:
        if col not in d.columns:
            d[col] = ''
        if treat_missing_as_category:
            d[col] = d[col].fillna('__MISSING__').astype(str)
        else:
            d[col] = d[col].astype(str).fillna('')

    encoded = pd.get_dummies(d, columns=categorical_columns, drop_first=drop_first, prefix_sep='=')
    encoded_columns = list(encoded.columns)
    return encoded, encoded_columns


def label_encode_boolean(df: pd.DataFrame, boolean_columns: List[str]) -> pd.DataFrame:
    d = df.copy()
    for col in boolean_columns:
        if col not in d.columns:
            d[col] = False
        d[col] = d[col].astype(bool).astype(int)
    return d

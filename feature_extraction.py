
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def load_components(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    # Ensure expected columns exist; coerce types for numeric fields
    expected = ['component_id', 'role', 'os_type', 'layer', 'criticality', 'exposure_level', 'is_patched']
    for c in expected:
        if c not in df.columns:
            df[c] = ''
    # cast criticality to numeric (coerce errors -> NaN)
    df['criticality'] = pd.to_numeric(df['criticality'], errors='coerce')
    # normalize is_patched
    df['is_patched'] = df['is_patched'].astype(str).str.lower().isin(['1','true','yes','y','t'])
    return df[expected]


def load_dependencies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    expected = ['source_component', 'target_component', 'dependency_type']
    for c in expected:
        if c not in df.columns:
            df[c] = ''
    return df[expected]


def load_vulnerabilities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    expected = ['component_id', 'cvss_score', 'attack_surface', 'access_vector']
    for c in expected:
        if c not in df.columns:
            df[c] = ''
    df['cvss_score'] = pd.to_numeric(df['cvss_score'], errors='coerce')
    return df[expected]


def aggregate_vuln_stats(vulns: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return mapping: component_id -> stats dict (vuln_count, base_cvss_sum, max_cvss, mean_cvss)"""
    out = {}
    if vulns.empty:
        return out
    grp = vulns.groupby('component_id')['cvss_score']
    for cid, series in grp:
        vals = series.dropna().astype(float).tolist()
        out[cid] = {
            'vuln_count': len(vals),
            'base_cvss_sum': float(sum(vals)) if vals else 0.0,
            'max_cvss': float(max(vals)) if vals else 0.0,
            'mean_cvss': float(np.mean(vals)) if vals else 0.0
        }
    return out


def compute_degree_features(deps: pd.DataFrame, component_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute in-degree and out-degree (counting all dependency types)."""
    out = {cid: {'in_degree': 0, 'out_degree': 0} for cid in component_ids}
    for _, row in deps.iterrows():
        src = str(row.get('source_component', '')).strip()
        tgt = str(row.get('target_component', '')).strip()
        if src in out:
            out[src]['out_degree'] += 1
        if tgt in out:
            out[tgt]['in_degree'] += 1
    return out


def extract_features_from_csvs(components_csv: str, dependencies_csv: str, vulnerabilities_csv: str,
                               fill_missing_criticality: Optional[int] = 1) -> pd.DataFrame:

    comps = load_components(components_csv)
    deps = load_dependencies(dependencies_csv)
    vulns = load_vulnerabilities(vulnerabilities_csv)

    component_ids = comps['component_id'].tolist()
    degree_map = compute_degree_features(deps, component_ids)
    vuln_map = aggregate_vuln_stats(vulns)

    rows = []
    for _, r in comps.iterrows():
        cid = r['component_id']
        criticality = r['criticality']
        if pd.isna(criticality):
            criticality = fill_missing_criticality
        row = {
            'component_id': cid,
            'role': r['role'],
            'os_type': r['os_type'],
            'layer': r['layer'],
            'exposure_level': r['exposure_level'],
            'is_patched': bool(r['is_patched']),
            'criticality': float(criticality),
            'in_degree': degree_map.get(cid, {}).get('in_degree', 0),
            'out_degree': degree_map.get(cid, {}).get('out_degree', 0),
        }
        v = vuln_map.get(cid, {'vuln_count': 0, 'base_cvss_sum': 0.0, 'max_cvss': 0.0, 'mean_cvss': 0.0})
        row.update(v)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('component_id', drop=False)
    # Ensure numeric columns exist
    numeric_cols = ['criticality','in_degree','out_degree','vuln_count','base_cvss_sum','max_cvss','mean_cvss']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df.reset_index(drop=True)

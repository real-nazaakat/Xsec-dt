
from typing import Dict, Any, Iterable, List, Tuple
import math

# Default weights (document these in Methods)
DEFAULT_WEIGHTS = {
    'w_c': 0.5,   # criticality weight
    'w_v': 0.3,   # vulnerability (cvss) weight
    'w_e': 0.15,  # exposure weight
    'w_p': 0.05   # patch / mitigation gap weight
}

EXPOSURE_MAP = {
    'isolated': 0.0,
    'internal': 0.25,
    'dmz': 0.6,
    'internet-facing': 1.0
}

def _norm_criticality(criticality: float) -> float:
    # criticality expected 1..5
    return max(0.0, min(1.0, (float(criticality) - 1.0) / 4.0))

def _norm_cvss_sum(cvss_sum: float, scale: float = 30.0) -> float:
    return max(0.0, min(1.0, float(cvss_sum) / float(scale)))

def _exposure_score(exposure: str) -> float:
    return float(EXPOSURE_MAP.get(str(exposure).lower(), 0.0))

def compute_node_risk(node_attrs: Dict[str, Any],
                      vuln_summary: Dict[str, Any],
                      weights: Dict[str, float] = None) -> Dict[str, float]:

    if weights is None:
        weights = DEFAULT_WEIGHTS
    crit = _norm_criticality(node_attrs.get('criticality', 1.0))
    base_cvss = float(vuln_summary.get('base_cvss_sum', 0.0))
    cvss = _norm_cvss_sum(base_cvss)
    exposure = _exposure_score(node_attrs.get('exposure_level', 'internal'))
    patched_flag = 1.0 if bool(node_attrs.get('is_patched', False)) else 0.0

    # Node risk: weighted sum
    node_risk = (weights['w_c'] * crit +
                 weights['w_v'] * cvss +
                 weights['w_e'] * exposure +
                 weights['w_p'] * (1.0 - patched_flag))

    # clamp
    node_risk = max(0.0, min(1.0, node_risk))
    return {
        'crit_norm': round(crit, 4),
        'cvss_norm': round(cvss, 4),
        'exposure': round(exposure, 4),
        'patched_flag': int(patched_flag),
        'node_risk': round(node_risk, 4),
        'base_cvss_sum': base_cvss,
        'vuln_count': int(vuln_summary.get('vuln_count', 0))
    }

def estimate_risk_reduction_if_patched(node_attrs: Dict[str, Any],
                                       vuln_summary: Dict[str, Any],
                                       weights: Dict[str, float] = None,
                                       patch_effectiveness: float = 0.6) -> Dict[str, Any]:

    current = compute_node_risk(node_attrs, vuln_summary, weights)

    patched_attrs = dict(node_attrs)
    patched_attrs['is_patched'] = True
    patched_vuln = dict(vuln_summary)
    patched_vuln['base_cvss_sum'] = patched_vuln.get('base_cvss_sum', 0.0) * patch_effectiveness
    post = compute_node_risk(patched_attrs, patched_vuln, weights)
    abs_red = round(current['node_risk'] - post['node_risk'], 4)
    rel_red = round(abs_red / (current['node_risk'] + 1e-12), 4) if current['node_risk'] > 0 else 0.0
    return {
        'current_risk': current['node_risk'],
        'post_patch_risk': post['node_risk'],
        'absolute_reduction': abs_red,
        'relative_reduction': rel_red
    }


from typing import List, Dict, Any, Tuple
from risk_scoring import estimate_risk_reduction_if_patched, compute_node_risk

EPS = 1e-6

def rank_by_roi(components: List[Dict[str, Any]],
                vuln_map: Dict[str, Dict[str, Any]],
                cost_map: Dict[str, float],
                weights: Dict[str, float] = None,
                patch_effectiveness: float = 0.6,
                top_k: int = None) -> List[Dict[str, Any]]:

    rows = []
    for comp in components:
        cid = comp.get('component_id')
        v = vuln_map.get(cid, {'base_cvss_sum': 0.0, 'vuln_count': 0})
        cost = float(cost_map.get(cid, 0.0))
        est = estimate_risk_reduction_if_patched(comp, v, weights=weights, patch_effectiveness=patch_effectiveness)
        abs_red = float(est['absolute_reduction'])
        roi = abs_red / (cost + EPS)
        rows.append({
            'component_id': cid,
            'current_risk': est['current_risk'],
            'post_patch_risk': est['post_patch_risk'],
            'absolute_reduction': abs_red,
            'cost': cost,
            'roi': roi,
            'vuln_count': int(v.get('vuln_count', 0))
        })
    rows.sort(key=lambda r: (-r['roi'], -r['absolute_reduction'], -r['current_risk']))
    if top_k is not None:
        return rows[:top_k]
    return rows

def rank_by_absolute_risk(components: List[Dict[str, Any]],
                          vuln_map: Dict[str, Dict[str, Any]],
                          weights: Dict[str, float] = None,
                          top_k: int = None) -> List[Dict[str, Any]]:
    rows = []
    for comp in components:
        cid = comp.get('component_id')
        v = vuln_map.get(cid, {'base_cvss_sum': 0.0, 'vuln_count': 0})
        r = compute_node_risk(comp, v, weights)
        rows.append({
            'component_id': cid,
            'current_risk': r['node_risk'],
            'vuln_count': int(v.get('vuln_count', 0))
        })
    rows.sort(key=lambda r: (-r['current_risk'], -r['vuln_count']))
    if top_k is not None:
        return rows[:top_k]
    return rows

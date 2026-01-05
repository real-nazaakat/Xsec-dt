
from typing import Dict, Any, Iterable, List


def compute_node_risk(component_attrs: Dict[str, Any],
                      vulnerabilities: Iterable[Dict[str, Any]]) -> Dict[str, float]:

    criticality = float(component_attrs.get("criticality", 1))
    patched = bool(component_attrs.get("is_patched", False))

    base, effective = 0.0, 0.0
    for v in vulnerabilities:
        cvss = float(v.get("cvss", 0.0))
        base += cvss
        effective += cvss * (0.6 if patched else 1.0)

    crit_norm = (criticality - 1) / 4.0
    cvss_norm = min(effective / 30.0, 1.0)

    return {
        "base_cvss_sum": base,
        "effective_cvss_sum": effective,
        "node_risk_score": round(0.6 * crit_norm + 0.4 * cvss_norm, 4),
    }


def aggregate_graph_risk(node_vuln_map, node_attrs_map):
    return {
        nid: compute_node_risk(node_attrs_map[nid], node_vuln_map.get(nid, []))
        for nid in node_attrs_map
    }


def propagate_risk_simple(graph, node_risk_map):
    out = {}
    for n, r in node_risk_map.items():
        preds = getattr(graph, "predecessors", lambda x: [])(n)
        mean_pred = sum(node_risk_map[p]["node_risk_score"] for p in preds) / len(preds) if preds else 0.0
        out[n] = round(min(1.0, r["node_risk_score"] + 0.5 * mean_pred), 4)
    return out

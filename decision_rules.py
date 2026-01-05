
from typing import Dict, Any, List

RULE_THRESHOLDS = {
    'urgent': 0.8,
    'high': 0.6,
    'medium': 0.3
}

def triage_rule(node_risk: float, vuln_count: int) -> str:
    if node_risk >= RULE_THRESHOLDS['urgent'] and vuln_count >= 1:
        return 'URGENT'
    if node_risk >= RULE_THRESHOLDS['high']:
        return 'HIGH'
    if node_risk >= RULE_THRESHOLDS['medium']:
        return 'MEDIUM'
    return 'LOW'

def rule_justification(component_id: str, node_risk: float, vuln_count: int, cost: float = None) -> str:
    level = triage_rule(node_risk, vuln_count)
    parts = [f'Component {component_id}: priority={level}.']
    parts.append(f'Computed node risk={node_risk:.3f}; vulnerabilities={vuln_count}.')
    if level == 'URGENT':
        parts.append('Recommendation: immediate patch or deploy compensating controls (isolation, network rules).')
    elif level == 'HIGH':
        parts.append('Recommendation: schedule patch within 7 days and monitor impact.')
    elif level == 'MEDIUM':
        parts.append('Recommendation: schedule patch within 30 days or monitor for exploitation signals.')
    else:
        parts.append('Recommendation: patch during the next maintenance window.')
    if cost is not None:
        parts.append(f'Estimated patch cost: {cost:.2f}.')
    return ' '.join(parts)

def decision_table_example() -> List[Dict[str, Any]]:
    return [
        {'priority': 'URGENT', 'condition': 'node_risk >= 0.8 AND vuln_count >= 1', 'action': 'Immediate patch / isolate'},
        {'priority': 'HIGH', 'condition': '0.6 <= node_risk < 0.8', 'action': 'Schedule patch within 7 days; increase monitoring'},
        {'priority': 'MEDIUM', 'condition': '0.3 <= node_risk < 0.6', 'action': 'Schedule patch within 30 days or monitor'},
        {'priority': 'LOW', 'condition': 'node_risk < 0.3', 'action': 'Patch in regular maintenance'}
    ]

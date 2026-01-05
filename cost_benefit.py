from typing import Dict, Any
import math

def compute_expected_loss(node_risk: float,
                          asset_value: float,
                          time_horizon_years: float = 1.0,
                          annual_prob_scale: float = 1.0) -> float:

    annual_p = max(0.0, min(1.0, node_risk * annual_prob_scale))
    # probability of at least one compromise within horizon
    p_horizon = 1.0 - (1.0 - annual_p) ** float(time_horizon_years)
    expected_loss = p_horizon * float(asset_value)
    return expected_loss

def compute_roi_from_patch(absolute_risk_reduction: float,
                           asset_value: float,
                           patch_cost: float,
                           time_horizon_years: float = 1.0,
                           annual_prob_scale: float = 1.0) -> Dict[str, float]:

    node_risk_before = absolute_risk_reduction + 0.0  

    benefit = compute_expected_loss(absolute_risk_reduction, asset_value, time_horizon_years, annual_prob_scale)
    roi = benefit / (patch_cost + 1e-9)
    payback = (patch_cost / benefit) if benefit > 0 else float('inf')
    return {
        'expected_benefit': benefit,
        'roi': roi,
        'payback_years': payback
    }

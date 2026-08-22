"""
trust_utils.py
Shared Trust Score logic for XrayTrust project.
Use the SAME file in both the research notebook (for evaluation/validation)
and the Streamlit app (for live demo), so the scoring logic never diverges.

Calibration basis (empirically derived from per-image evaluation results):
    SSIM range: ~0.70 to 0.88
    Physics Error range: 0.01 to 0.15
    L1 Error range: 0.015 to 0.16 (Per-image scale)
"""

# ---- Normalization bounds (Derived from per-image evaluation scale) ----
SSIM_LO, SSIM_HI = 0.70, 0.88
PHYSICS_ERROR_LO, PHYSICS_ERROR_HI = 0.01, 0.15
L1_ERROR_LO, L1_ERROR_HI = 0.015, 0.16

# ---- Weights for combining into a single score ----
WEIGHTS = {
    "ssim": 0.35,
    "physics": 0.45,
    "l1": 0.20,
}

# ---- Status thresholds on the final 0-100 trust score ----
STATUS_HIGH_MIN = 75
STATUS_MEDIUM_MIN = 50


def _normalize_ssim(ssim_val: float) -> float:
    """Map SSIM to a 0-1 'goodness' score using the observed experiment range."""
    score = (ssim_val - SSIM_LO) / (SSIM_HI - SSIM_LO)
    return max(0.0, min(1.0, score))


def _normalize_physics_error(err: float) -> float:
    """Lower physics error = better. Map to 0-1 goodness score (inverted)."""
    score = 1.0 - (err - PHYSICS_ERROR_LO) / (PHYSICS_ERROR_HI - PHYSICS_ERROR_LO)
    return max(0.0, min(1.0, score))


def _normalize_l1_error(err: float) -> float:
    """Lower L1 pixel error = better. Map to 0-1 goodness score (inverted)."""
    score = 1.0 - (err - L1_ERROR_LO) / (L1_ERROR_HI - L1_ERROR_LO)
    return max(0.0, min(1.0, score))


def calculate_trust_score(ssim_val: float = None, physics_error: float = 0.0, l1_error: float = None):
    """
    Compute a single Trust Score (0-100) and an inspection-readiness status.

    Parameters
    ----------
    ssim_val : float, optional
        Structural similarity between generated HR and ground-truth HR.
    physics_error : float
        L1 error between (downsample->upsample of generated HR) and original LR input.
    l1_error : float, optional
        Direct pixel-level L1 error vs ground truth.

    Returns
    -------
    dict with keys: trust_score (0-100), status, recommendation, breakdown
    """
    have_gt = ssim_val is not None and l1_error is not None

    physics_score = _normalize_physics_error(physics_error)

    if have_gt:
        ssim_score = _normalize_ssim(ssim_val)
        l1_score = _normalize_l1_error(l1_error)
        trust = (
            WEIGHTS["ssim"] * ssim_score
            + WEIGHTS["physics"] * physics_score
            + WEIGHTS["l1"] * l1_score
        )
    else:
        # Ground Truth ছাড়া শুধু Physics Error ব্যবহার হবে
        trust = physics_score

    trust_pct = round(trust * 100, 2)

    if trust_pct >= STATUS_HIGH_MIN:
          status = "HIGH"
          recommendation = "High reconstruction consistency"
    elif trust_pct >= STATUS_MEDIUM_MIN:
          status = "MEDIUM"
          recommendation = "Review reconstruction carefully"
    else:
       status = "LOW"
       recommendation = "Low reconstruction consistency"
    
    return {
        "trust_score": trust_pct,
        "status": status,
        "recommendation": recommendation,
        "physics_error": physics_error,
        "ssim": ssim_val,
        "l1_error": l1_error,
        "ground_truth_used": have_gt,
    }


if __name__ == "__main__":
    # Test with 40-shot values: SSIM = 0.8391116287, Physics = 0.01707893554, L1 = 0.0206435604
    res = calculate_trust_score(ssim_val=0.8391116287, physics_error=0.01707893554, l1_error=0.0206435604)
    print("40-shot validation test:")
    print(res)
    # Expected Trust Score: ~89.15%
"""
EXPIRYRANGE SIGNAL ENGINE — 1HZ10V
====================================
Four models. Each is genuinely different — not four ways of saying
the same thing.

MODEL A — Fat-Tail Adjusted Normal Distribution
  What it measures: statistical probability that price stays inside ±1.6
  at expiry, using EWMA volatility (responds in ~5 ticks to vol changes)
  and fat-tail correction via excess kurtosis.

  KEY FIX from trade data analysis:
    When normal_dist:OK fires ALONE (bollinger:NO, ou:NO), win rate was
    28.6% across 14 trades. This means the Gaussian model systematically
    overestimates safety on 1HZ10V during certain conditions.
    Fix: normal_dist requires a HIGHER standalone probability threshold
    (0.72 vs 0.58) if it is the only vol model passing. When bollinger
    or OU also pass, the standard 0.58 threshold applies.

MODEL B — Bollinger Band Compression + Velocity
  What it measures: is the market coiled (low vol relative to its own
  history)? Is price sitting near the midline? Is it moving fast?
  Uses percentile rank compression — scale-independent.

MODEL C — Ornstein-Uhlenbeck Process (on de-meaned returns)
  What it measures: speed of mean reversion. A fast reverting process
  (low half-life) will snap back inside barriers before expiry.
  Fits on de-meaned returns to eliminate price-level bias in OLS.
  Checks stability across two sub-windows — unstable theta = noisy fit.

MODEL D — Tick Autocorrelation
  What it measures: is price oscillating (each tick partially reverses
  the previous) or trending (each tick continues the previous)?
  Negative lag-1 ACF = oscillating = safe for EXPIRYRANGE.
  This is ORTHOGONAL to A/B/C — can detect trending at low vol
  or oscillating at high vol. Neither case is visible to the others.

CONFLUENCE: 2-of-4 + avg_conf ≥ 0.55 + top_conf ≥ 0.70
  Autocorr agreement adds +0.05 confidence bonus (independent signal).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from scipy.stats import norm as scipy_norm

import settings as S


@dataclass
class ModelResult:
    name:       str
    tradeable:  bool
    confidence: float
    detail:     Dict  = field(default_factory=dict)


@dataclass
class ERSignal:
    tradeable:   bool
    confidence:  float
    score:       int          # how many models voted
    reasons:     list
    models:      Dict         = field(default_factory=dict)
    skip_reason: str          = ""


# ─────────────────────────────────────────────────────────────────
# MODEL A — Fat-Tail Adjusted Normal Distribution
# ─────────────────────────────────────────────────────────────────

def _ewma_vol(rets: np.ndarray, halflife: int = 20) -> float:
    """EWMA volatility — weights recent ticks exponentially."""
    alpha    = 1.0 - np.exp(-np.log(2) / halflife)
    var      = float(rets[-1] ** 2)
    for r in reversed(rets[:-1]):
        var  = alpha * r**2 + (1.0 - alpha) * var
    return float(np.sqrt(var))


def model_normal_dist(prices: np.ndarray, barrier: float,
                      expiry_ticks: int) -> ModelResult:
    name = "normal_dist"
    fail = ModelResult(name, False, 0.0)

    if len(prices) < 25:
        return fail

    rets = np.diff(prices[-120:]) if len(prices) >= 121 else np.diff(prices)
    if len(rets) < 10:
        return fail

    sigma = _ewma_vol(rets)
    if sigma == 0:
        return fail

    sigma_T = sigma * np.sqrt(expiry_ticks)
    prob    = float(scipy_norm.cdf(barrier / sigma_T) -
                    scipy_norm.cdf(-barrier / sigma_T))

    # ── Fat-tail correction ───────────────────────────────────────
    excess_kurt = 0.0
    if len(rets) >= 20:
        mu_r  = float(np.mean(rets))
        sd_r  = float(np.std(rets, ddof=1)) or 1e-8
        kurt  = float(np.mean(((rets - mu_r) / sd_r) ** 4))
        excess_kurt = max(0.0, kurt - 3.0)
        prob *= float(np.clip(1.0 - 0.04 * excess_kurt, 0.7, 1.0))

    # ── Spike gate ───────────────────────────────────────────────
    recent_abs = np.abs(np.diff(prices[-21:])) if len(prices) >= 22 else np.abs(rets)
    max_tick   = float(recent_abs.max()) if len(recent_abs) else 0.0
    spike_risk = max_tick > barrier * 0.4

    # ── Vol expansion gate ───────────────────────────────────────
    vol_expanding = False
    if len(prices) >= 55:
        ra = np.abs(np.diff(prices))
        vol_expanding = float(np.std(ra[-10:], ddof=1)) > float(np.std(ra[-50:-10], ddof=1)) * 1.5

    tradeable = (prob >= S.NORMAL_DIST_MIN_PROB
                 and not spike_risk
                 and not vol_expanding)

    confidence = float(np.clip(
        (prob - S.NORMAL_DIST_MIN_PROB) / (1.0 - S.NORMAL_DIST_MIN_PROB + 1e-6),
        0.0, 1.0))

    return ModelResult(name, tradeable, confidence, {
        "prob":         round(prob, 4),
        "sigma_T":      round(sigma_T, 5),
        "excess_kurt":  round(excess_kurt, 3),
        "spike_risk":   spike_risk,
        "vol_expanding": vol_expanding,
        "standalone_threshold": S.NORMAL_DIST_STANDALONE_PROB,
    })


# ─────────────────────────────────────────────────────────────────
# MODEL B — Bollinger Band Compression + Velocity
# ─────────────────────────────────────────────────────────────────

def model_bollinger(prices: np.ndarray, period: int = 20) -> ModelResult:
    name = "bollinger"
    fail = ModelResult(name, False, 0.0)

    if len(prices) < period + 10:
        return fail

    arr   = prices[-period:]
    mid   = float(np.mean(arr))
    std   = float(np.std(arr, ddof=1))
    if mid == 0 or std == 0:
        return fail

    upper = mid + 2 * std
    lower = mid - 2 * std
    price = float(prices[-1])
    width = (2 * 2 * std) / mid
    pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5

    # ── Relative compression ──────────────────────────────────────
    hist_len = min(len(prices), 400)
    segs     = np.arange(period, hist_len, max(1, hist_len // 60))
    hist_w   = []
    for i in segs:
        seg = prices[-i - period:-i] if i > 0 else prices[-period:]
        if len(seg) < period:
            continue
        sm = float(np.mean(seg))
        ss = float(np.std(seg, ddof=1))
        if sm > 0:
            hist_w.append((4 * ss) / sm)

    if len(hist_w) >= 8:
        ha    = np.array(hist_w)
        rank  = float(np.sum(ha <= width) / len(ha))
        ratio = float(ha.max() / (ha.min() + 1e-12))
        compressed = (ratio < 2.0) or (rank <= S.BB_COMPRESSION_PCT)
    else:
        compressed = width < 0.015

    near_mid = S.BB_MIDZONE_LO <= pct_b <= S.BB_MIDZONE_HI

    # ── Velocity gate ─────────────────────────────────────────────
    vel_danger = False
    if len(prices) >= 6:
        vel_danger = abs(float(prices[-1]) - float(prices[-6])) / (std + 1e-8) > 1.5

    # ── Trend pressure ────────────────────────────────────────────
    trend_pressure = False
    if len(prices) >= 11:
        moves = np.diff(prices[-11:])
        frac  = float(np.sum(moves > 0)) / len(moves)
        trend_pressure = frac >= 0.7 or frac <= 0.3

    tradeable = compressed and near_mid and not vel_danger and not trend_pressure

    conf = 0.0
    if tradeable and hist_w:
        ha        = np.array(hist_w)
        comp_scr  = 1.0 - float(np.sum(ha <= width) / len(ha))
        mid_scr   = 1.0 - 2 * abs(pct_b - 0.5)
        conf      = float((comp_scr + mid_scr) / 2.0)

    return ModelResult(name, tradeable, conf, {
        "width":          round(width, 6),
        "pct_b":          round(pct_b, 4),
        "compressed":     compressed,
        "near_mid":       near_mid,
        "vel_danger":     vel_danger,
        "trend_pressure": trend_pressure,
    })


# ─────────────────────────────────────────────────────────────────
# MODEL C — Ornstein-Uhlenbeck (de-meaned returns)
# ─────────────────────────────────────────────────────────────────

def _fit_ou(X: np.ndarray):
    """OLS fit of OU on array X. Returns (theta, half_life)."""
    Xlag = X[:-1]
    Xnow = X[1:]
    A    = np.vstack([np.ones(len(Xlag)), Xlag]).T
    c, _, _, _ = np.linalg.lstsq(A, Xnow, rcond=None)
    theta = 1.0 - float(c[1])
    if theta <= 0:
        return None, None
    return theta, float(np.log(2) / theta)


def model_ou(prices: np.ndarray, expiry_minutes: float) -> ModelResult:
    name = "ou"
    fail = ModelResult(name, False, 0.0)

    n = len(prices)
    if n < 40:
        return fail

    try:
        window  = prices[-150:] if n >= 150 else prices
        rets    = np.diff(window)
        X       = rets - float(np.mean(rets))   # de-meaned

        theta, hl = _fit_ou(X)
        if theta is None:
            return fail

        max_hl  = expiry_minutes * S.OU_HALFLIFE_MAX_RATIO * 60

        # ── Theta stability across two halves ─────────────────────
        mid      = len(X) // 2
        t1, _    = _fit_ou(X[:mid])
        t2, _    = _fit_ou(X[mid:])
        stable   = False
        if t1 and t2 and t1 > 0 and t2 > 0:
            ratio  = max(t1, t2) / (min(t1, t2) + 1e-8)
            stable = ratio < S.OU_THETA_STABILITY_RATIO

        # ── OU residual normality check ───────────────────────────
        # If residuals are extremely fat-tailed the OU fit is unreliable
        residuals = X[1:] - (X[:-1] * (1.0 - theta))
        res_kurt  = 0.0
        if len(residuals) >= 10:
            rs    = float(np.std(residuals, ddof=1)) or 1e-8
            res_kurt = float(np.mean(((residuals - residuals.mean()) / rs) ** 4))

        tradeable = (
            hl < max_hl
            and stable
            and theta >= S.OU_MIN_THETA
            and res_kurt < 20.0    # reject wildly non-Gaussian residuals
        )

        confidence = float(np.clip(1.0 - hl / (max_hl + 1e-6), 0.0, 1.0))
        if not stable:
            confidence = 0.0

        return ModelResult(name, tradeable, confidence, {
            "theta":      round(theta, 6),
            "half_life":  round(hl, 3),
            "max_hl":     round(max_hl, 1),
            "stable":     stable,
            "res_kurt":   round(res_kurt, 2),
        })

    except Exception:
        return fail


# ─────────────────────────────────────────────────────────────────
# MODEL D — Tick Autocorrelation
# ─────────────────────────────────────────────────────────────────

def _acf(r: np.ndarray, lag: int) -> float:
    if len(r) <= lag:
        return 0.0
    mu  = float(np.mean(r))
    var = float(np.var(r, ddof=1))
    if var == 0:
        return 0.0
    return float(np.mean((r[:-lag] - mu) * (r[lag:] - mu)) / var)


def model_autocorr(prices: np.ndarray) -> ModelResult:
    name = "autocorr"
    fail = ModelResult(name, False, 0.0)

    if len(prices) < 40:
        return fail

    rets = np.diff(prices[-120:] if len(prices) >= 120 else prices)
    if len(rets) < 20:
        return fail

    a1 = _acf(rets, 1)
    a2 = _acf(rets, 2)

    # Consistency across two sub-windows — not just a snapshot
    half       = len(rets) // 2
    a1_h1      = _acf(rets[:half], 1)
    a1_h2      = _acf(rets[half:], 1)
    consistent = (a1_h1 < 0 and a1_h2 < 0)

    tradeable = (
        a1  < S.ACF_LAG1_THRESHOLD
        and consistent
        and a2  < S.ACF_LAG2_MAX
    )

    confidence = float(np.clip(-a1 * 2.0, 0.0, 1.0)) if tradeable else 0.0

    return ModelResult(name, tradeable, confidence, {
        "acf1":       round(a1, 5),
        "acf2":       round(a2, 5),
        "acf1_h1":    round(a1_h1, 5),
        "acf1_h2":    round(a1_h2, 5),
        "consistent": consistent,
    })


# ─────────────────────────────────────────────────────────────────
# VOL STABILITY PRE-FILTER
# ─────────────────────────────────────────────────────────────────

def vol_is_stable(prices: np.ndarray) -> tuple:
    """
    Check whether EWMA volatility has been stable over the last
    VOL_STABILITY_TICKS ticks. If vol spiked recently the market
    is in a transitional state — not safe for EXPIRYRANGE.

    Returns (stable: bool, reason: str)
    """
    n = len(prices)
    needed = S.VOL_STABILITY_TICKS + 10
    if n < needed:
        return True, ""   # not enough data to check — allow through

    def ewma_vol(p: np.ndarray) -> float:
        rets  = np.diff(p[-40:]) if len(p) >= 41 else np.diff(p)
        if len(rets) == 0: return 0.0
        alpha = 1.0 - np.exp(-np.log(2) / 20.0)
        var   = float(rets[-1] ** 2)
        for r in reversed(rets[:-1]):
            var = alpha * r**2 + (1.0 - alpha) * var
        return float(np.sqrt(var))

    vol_now  = ewma_vol(prices)
    vol_then = ewma_vol(prices[:-S.VOL_STABILITY_TICKS])

    if vol_then == 0:
        return True, ""

    change = abs(vol_now - vol_then) / vol_then
    if change > S.VOL_STABILITY_THRESHOLD:
        return False, f"VOL_UNSTABLE {change:.1%} change in {S.VOL_STABILITY_TICKS} ticks"

    return True, ""


# ─────────────────────────────────────────────────────────────────
# CONFLUENCE ENGINE
# ─────────────────────────────────────────────────────────────────

def evaluate(prices: np.ndarray) -> ERSignal:
    """
    Run all 4 models and produce a single tradeable/not decision.

    Normal-dist standalone fix:
      If normal_dist passes but NEITHER bollinger NOR ou passes,
      require its prob to exceed NORMAL_DIST_STANDALONE_PROB (0.72)
      instead of the standard 0.58. This eliminates the 28.6% WR
      pattern seen in live trade data where nD:OK alone fires.
    """
    barrier      = S.BARRIER_VALUE
    expiry_ticks = S.EXPIRY_TICKS
    expiry_mins  = S.EXPIRY_MINUTES

    # Vol stability pre-filter — before running any model
    stable, reason = vol_is_stable(prices)
    if not stable:
        return ERSignal(
            tradeable=False, confidence=0.0, score=0,
            reasons=[], skip_reason=reason
        )

    mA = model_normal_dist(prices, barrier, expiry_ticks)
    mB = model_bollinger(prices)
    mC = model_ou(prices, expiry_mins)
    mD = model_autocorr(prices)

    all_models  = {"normal_dist": mA, "bollinger": mB, "ou": mC, "autocorr": mD}
    passing     = {k: v for k, v in all_models.items() if v.tradeable}
    score       = len(passing)

    # ── Normal-dist standalone penalty ───────────────────────────
    # From live data: nD:OK alone = 28.6% WR. The fix is to apply a
    # much higher probability gate when nD is the only vol model voting.
    if (mA.tradeable
            and not mB.tradeable
            and not mC.tradeable):
        # Only mA (and possibly mD) passing — require higher prob
        actual_prob = mA.detail.get("prob", 0.0)
        if actual_prob < S.NORMAL_DIST_STANDALONE_PROB:
            # Remove mA from passing set — doesn't meet standalone bar
            passing.pop("normal_dist", None)
            score = len(passing)

    if score < S.MODELS_REQUIRED:
        names = [f"{k}:{'OK' if v.tradeable else 'NO'}({v.confidence:.2f})"
                 for k, v in all_models.items()]
        return ERSignal(
            tradeable=False, confidence=0.0, score=score,
            reasons=names,
            models={k: v.detail for k, v in all_models.items()},
            skip_reason=f"MODELS_DISAGREE {score}/{len(all_models)}"
        )

    confs      = [v.confidence for v in passing.values()]
    confidence = float(np.mean(confs))

    # ── ACF bonus — orthogonal confirmation ───────────────────────
    if mD.tradeable:
        confidence = min(1.0, confidence + 0.05)

    if confidence < S.CONFIDENCE_MIN:
        names = [f"{k}:{'OK' if v.tradeable else 'NO'}({v.confidence:.2f})"
                 for k, v in all_models.items()]
        return ERSignal(
            tradeable=False, confidence=confidence, score=score,
            reasons=names,
            models={k: v.detail for k, v in all_models.items()},
            skip_reason=f"LOW_CONF {confidence:.3f}<{S.CONFIDENCE_MIN}"
        )

    top_conf = max(v.confidence for v in passing.values())
    if top_conf < S.TOP_CONF_MIN:
        names = [f"{k}:{'OK' if v.tradeable else 'NO'}({v.confidence:.2f})"
                 for k, v in all_models.items()]
        return ERSignal(
            tradeable=False, confidence=confidence, score=score,
            reasons=names,
            models={k: v.detail for k, v in all_models.items()},
            skip_reason=f"NO_STRONG_MODEL top={top_conf:.3f}<{S.TOP_CONF_MIN}"
        )

    reasons = [f"{k}:{'OK' if v.tradeable else 'NO'}({v.confidence:.2f})"
               for k, v in all_models.items()]
    return ERSignal(
        tradeable=True, confidence=round(confidence, 4),
        score=score, reasons=reasons,
        models={k: v.detail for k, v in all_models.items()}
    )

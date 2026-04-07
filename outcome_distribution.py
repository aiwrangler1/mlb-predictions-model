"""
Helpers for sampling and summarizing DFS outcome distributions.

The goal is to turn a sparse percentile ladder into a usable inverse-CDF
sampler so lineup simulation can respect asymmetry and tail risk better than
a single Gaussian approximation.
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, Mapping, Sequence


def _clean_points(points: Mapping[float, float], floor: float = 0.0) -> list[tuple[float, float]]:
    cleaned = []
    for prob, value in points.items():
        if value is None:
            continue
        if prob < 0.0 or prob > 1.0:
            continue
        cleaned.append((float(prob), max(float(value), floor)))
    cleaned.sort(key=lambda item: item[0])
    return cleaned


def _enforce_monotonic(values: Sequence[float]) -> list[float]:
    result = []
    last = 0.0
    for value in values:
        current = max(float(value), last)
        result.append(current)
        last = current
    return result


def first_present_float(
    row: Mapping[str, object],
    keys: Sequence[str],
    default: float = 0.0,
    *,
    positive_only: bool = False,
) -> float:
    """
    Return the first numeric value found for any of the given keys.

    This lets loaders accept multiple schema variants without duplicating
    fallback logic across modules.
    """
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if positive_only and parsed <= 0:
            continue
        return parsed
    return default


def empirical_projection_profile(row: Mapping[str, object], fallback_mean: float = 0.0) -> Dict[str, float]:
    """
    Build a normalized player projection profile from any supported schema.

    Supports both the older DK export column names and the empirical sim
    columns emitted by the game model.
    """
    mean_proj = first_present_float(
        row,
        [
            "dk_mean",
            "dk_median",
            "dk_p50",
            "dk_50_percentile",
            "dk_points",
            "My Proj",
            "SS Proj",
            "Live Proj",
            "Actual",
        ],
        default=fallback_mean,
        positive_only=True,
    )
    median_proj = first_present_float(
        row,
        ["dk_median", "dk_p50", "dk_50_percentile"],
        default=mean_proj,
    )
    p10 = first_present_float(row, ["dk_p10", "dk_10_percentile"], default=median_proj)
    p25 = first_present_float(row, ["dk_p25", "dk_25_percentile"], default=median_proj)
    p50 = first_present_float(row, ["dk_p50", "dk_median", "dk_50_percentile"], default=median_proj)
    p75 = first_present_float(row, ["dk_p75", "dk_75_percentile"], default=median_proj)
    p85 = first_present_float(row, ["dk_p85", "dk_85_percentile"], default=median_proj)
    p90 = first_present_float(row, ["dk_p90", "dk_90_percentile"], default=median_proj)
    p95 = first_present_float(row, ["dk_p95", "dk_95_percentile"], default=median_proj)
    p99 = first_present_float(row, ["dk_p99", "dk_99_percentile"], default=median_proj)
    return {
        "mean": mean_proj,
        "median": median_proj,
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p85": p85,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }


def inverse_cdf_sample(points: Mapping[float, float], rng: random.Random, floor: float = 0.0) -> float:
    """
    Sample from a monotone inverse CDF defined by quantile anchors.

    Expected inputs are quantiles like:
        {0.25: p25, 0.50: p50, 0.75: p75, 0.85: p85, 0.95: p95, 0.99: p99}

    The function builds tail endpoints and then linearly interpolates between
    the anchors. This keeps the shape asymmetric and respects the supplied
    percentile ladder much better than a Gaussian fit.
    """
    cleaned = _clean_points(points, floor=floor)
    if not cleaned:
        return floor

    probs = [p for p, _ in cleaned]
    vals = [v for _, v in cleaned]

    # Build tail anchors if they are missing.
    if probs[0] > 0.0:
        if 0.25 in points and 0.50 in points:
            p25 = max(float(points[0.25]), floor)
            p50 = max(float(points[0.50]), p25)
            lower = max(floor, p25 - 1.5 * max(p50 - p25, 0.0))
        else:
            lower = max(floor, vals[0] * 0.5)
        probs.insert(0, 0.0)
        vals.insert(0, lower)

    if probs[-1] < 1.0:
        p95 = float(points.get(0.95, vals[-1]))
        p99 = float(points.get(0.99, vals[-1]))
        upper = max(vals[-1], p99 + 1.5 * max(p99 - p95, 0.0), p99 + max(0.5, 0.25 * max(p99, 1.0)))
        probs.append(1.0)
        vals.append(max(upper, vals[-1]))

    vals = _enforce_monotonic(vals)

    u = rng.random()
    for i in range(len(probs) - 1):
        left_p = probs[i]
        right_p = probs[i + 1]
        if u <= right_p or i == len(probs) - 2:
            left_v = vals[i]
            right_v = vals[i + 1]
            if right_p <= left_p:
                return left_v
            t = (u - left_p) / (right_p - left_p)
            return left_v + t * (right_v - left_v)

    return vals[-1]


def summarize_samples(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p85": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "std": 0.0,
        }

    ordered = sorted(float(x) for x in samples)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return ordered[idx]

    avg = mean(ordered)
    var = sum((x - avg) ** 2 for x in ordered) / n
    return {
        "mean": avg,
        "median": pct(50),
        "p10": pct(10),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p85": pct(85),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "std": math.sqrt(var),
    }

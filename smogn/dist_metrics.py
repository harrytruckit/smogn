# optimised_distances.py
# ---------------------------------------------------------------------------
# Fast, vectorised distance routines for SMOGN-style sampling
#
# Replaces: euclidean_dist, heom_dist, overlap_dist
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-30  # small constant to avoid divide-by-zero warnings


# ---------------------------------------------------------------------------
# Euclidean distance (numeric-only)
# ---------------------------------------------------------------------------
def euclidean_dist(a: pd.Series | np.ndarray,
                   b: pd.Series | np.ndarray,
                   d: int | None = None) -> float:
    """
    Vectorised Euclidean distance between two numeric 1-D vectors.
    `d` is accepted for API compatibility but ignored.

    Parameters
    ----------
    a, b : 1-D pandas.Series or 1-D numpy.ndarray
        Numeric feature vectors of equal length.

    Returns
    -------
    float
        Euclidean distance.
    """
    # zero-copy extraction of the underlying ndarray
    a_arr = a.to_numpy(copy=True) if isinstance(a, pd.Series) else np.asarray(a)
    b_arr = b.to_numpy(copy=True) if isinstance(b, pd.Series) else np.asarray(b)

    return float(np.linalg.norm(a_arr.astype(float) - b_arr.astype(float), ord=2))


# ---------------------------------------------------------------------------
# HEOM distance (mixed numeric + categorical)
# ---------------------------------------------------------------------------
def heom_dist(a_num: pd.Series | np.ndarray,
              b_num: pd.Series | np.ndarray,
              d_num: int,
              ranges_num: list | np.ndarray,
              a_nom: pd.Series | np.ndarray,
              b_nom: pd.Series | np.ndarray,
              d_nom: int) -> float:
    """
    Heterogeneous Euclidean–Overlap Metric (HEOM) distance.

    Numeric part is the (normalised) Euclidean distance; categorical part
    is the Hamming (overlap) distance.  Follows Wilson & Martinez (1997).

    All parameters keep the original order for full backwards compatibility.
    """
    # ----- numeric sub-distance ------------------------------------------------
    if d_num:
        a_n = a_num.to_numpy(copy=True) if isinstance(a_num, pd.Series) else np.asarray(a_num)
        b_n = b_num.to_numpy(copy=True) if isinstance(b_num, pd.Series) else np.asarray(b_num)

        a_n = a_n.astype(float, copy=True)
        b_n = b_n.astype(float, copy=True)

        ranges = np.asarray(ranges_num, dtype=float, copy=True)
        # Avoid division by zero by substituting 1 for degenerate ranges
        ranges_safe = np.where(ranges > _EPS, ranges, 1.0)

        diff_norm = (np.abs(a_n - b_n) / ranges_safe) ** 2
        dist_num = diff_norm.sum(dtype=float)
    else:          # no numeric features
        dist_num = 0.0

    # ----- categorical sub-distance -------------------------------------------
    if d_nom:
        a_c = a_nom.to_numpy(copy=True) if isinstance(a_nom, pd.Series) else np.asarray(a_nom)
        b_c = b_nom.to_numpy(copy=True) if isinstance(b_nom, pd.Series) else np.asarray(b_nom)

        dist_cat = np.count_nonzero(a_c != b_c)
    else:          # no categorical features
        dist_cat = 0

    # ----- combined HEOM distance ---------------------------------------------
    return float(np.sqrt(dist_num + dist_cat, dtype=float))


# ---------------------------------------------------------------------------
# Hamming / overlap distance (categorical-only)
# ---------------------------------------------------------------------------
def overlap_dist(a: pd.Series | np.ndarray,
                 b: pd.Series | np.ndarray,
                 d: int | None = None) -> float:
    """
    Hamming (overlap) distance between two categorical 1-D vectors.
    `d` is accepted for API compatibility but ignored.
    """
    a_arr = a.to_numpy(copy=True) if isinstance(a, pd.Series) else np.asarray(a)
    b_arr = b.to_numpy(copy=True) if isinstance(b, pd.Series) else np.asarray(b)

    return float(np.count_nonzero(a_arr != b_arr))

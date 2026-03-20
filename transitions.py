"""
transitions.py — Transition detection and statistics

Functions for:
  - Detecting regime transitions (A→B, B→A) in time series
  - Computing transition durations and return periods
  - CCDF analysis and exponential fitting
"""

import numpy as np
from scipy.stats import linregress


# Basin thresholds (dimensional, m/s)
VELOCITY_SCALE = 2.5e5 / (24 * 3600.0)   # ~2.89 m/s
U_A = 53.8    # m/s — strong vortex threshold
U_B = 1.75    # m/s — weak vortex threshold (paper Eq. 3)


def calculate_transition_durations(y_values, upper_bound, lower_bound):
    """
    Compute durations of B→A transitions in a 1-D zonal wind time series.

    A transition begins when the signal drops below lower_bound and ends when
    it rises above upper_bound.

    Parameters
    ----------
    y_values : array-like
        Time series of U at the reference level.
    upper_bound, lower_bound : float
        Thresholds defining regimes A and B.

    Returns
    -------
    list of int
        Duration (in time steps) of each detected transition.
    """
    durations = []
    transition_start = None
    above_upper = False
    below_lower = False

    for i in range(1, len(y_values)):
        if y_values[i] < lower_bound:
            below_lower = True
            above_upper = False
        elif y_values[i] > upper_bound:
            if below_lower and transition_start is not None:
                durations.append(i - transition_start)
                transition_start = None
            above_upper = True
            below_lower = False

        if below_lower and transition_start is None:
            transition_start = i

    return durations


def ccdf_slope(durations):
    """
    Compute the CCDF slope (exponential decay rate) of transition durations.

    Returns
    -------
    slope : float
        Best-fit slope of log(CCDF) vs duration.
    intercept : float
        Intercept of the linear fit.
    """
    if len(durations) < 2 or len(np.unique(durations)) < 2:
        return np.nan, np.nan

    data_sorted = np.sort(durations)
    ccdf = 1 - np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    valid = ccdf > 0
    slope, intercept, *_ = linregress(data_sorted[valid], np.log(ccdf[valid]))
    return slope, intercept


def exponential_fit_rate(durations):
    """Return the exponential fit rate 1/mean(durations)."""
    if len(durations) == 0:
        return np.nan
    return 1.0 / np.mean(durations)


def classify_states(u_timeseries, u_a=U_A, u_b=U_B):
    """
    Classify each time step into state A, B, or intermediate (C).

    Parameters
    ----------
    u_timeseries : array-like
        Zonal wind at the reference altitude (dimensional, m/s).
    u_a, u_b : float
        Thresholds for states A and B per Eqs. (2)-(3) in the paper.

    Returns
    -------
    labels : ndarray of str
        'A', 'B', or 'C' for each time step.
    """
    u = np.asarray(u_timeseries)
    labels = np.full(len(u), 'C', dtype='<U2')
    labels[u >= u_a] = 'A'
    labels[u <= u_b] = 'B'
    return labels


def detect_AB_transitions(u_timeseries, u_a=U_A, u_b=U_B):
    """
    Find indices of genuine A→B transitions (SSW events).

    A day t is an A→B transition point if:
      1. U(t) >= u_a
      2. U(t+1) < u_a
      3. The system reaches B before returning to A.

    Returns
    -------
    list of int
        Indices where A→B transitions begin.
    """
    u = np.asarray(u_timeseries)
    n = len(u)
    transitions = []

    i = 0
    while i < n - 1:
        if u[i] >= u_a and u[i + 1] < u_a:
            # Check if we reach B before A
            j = i + 1
            while j < n:
                if u[j] >= u_a:
                    break
                if u[j] <= u_b:
                    transitions.append(i)
                    break
                j += 1
            i = j
        else:
            i += 1

    return transitions


def detect_BA_transitions(u_timeseries, u_a=U_A, u_b=U_B):
    """
    Find indices of genuine B→A transitions (vortex recovery events).

    Analogous to detect_AB_transitions but in the reverse direction.
    """
    u = np.asarray(u_timeseries)
    n = len(u)
    transitions = []

    i = 0
    while i < n - 1:
        if u[i] <= u_b and u[i + 1] > u_b:
            j = i + 1
            while j < n:
                if u[j] <= u_b:
                    break
                if u[j] >= u_a:
                    transitions.append(i)
                    break
                j += 1
            i = j
        else:
            i += 1

    return transitions

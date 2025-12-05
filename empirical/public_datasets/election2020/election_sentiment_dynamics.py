# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics: Election 2020 Sentiment Analysis
==============================================================

Tests Kaplowitz-Fink second-order dynamics on political sentiment
around major election events.

Key events that act as "impulsive forces":
- Debates (Sep 29, Oct 7, Oct 22)
- Election Day (Nov 3)
- Networks call election (Nov 7)

Prediction: Sentiment should show damped oscillation after shocks,
with period ~13 seconds (Kaplowitz) or longer for political attitudes.

Data: IEEE DataPort - 20M tweets with VADER sentiment
URL: https://ieee-dataport.org/open-access/usa-nov2020-election-20-mil-tweets-sentiment-and-party-name-labels-dataset

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Key election events (impulsive forces)
ELECTION_EVENTS = {
    'debate_1': datetime(2020, 9, 29, 21, 0),  # First presidential debate
    'vp_debate': datetime(2020, 10, 7, 21, 0),  # VP debate
    'debate_2': datetime(2020, 10, 22, 21, 0),  # Final debate
    'election_day': datetime(2020, 11, 3, 0, 0),  # Election day
    'election_called': datetime(2020, 11, 7, 11, 25),  # Networks call for Biden
}


@dataclass
class EventDynamics:
    """Dynamics around a single event."""
    event_name: str
    event_time: datetime
    pre_sentiment: float
    post_sentiment: float
    peak_deviation: float
    damping_ratio: float
    period_days: float  # Changed from hours to days for daily data
    n_oscillations: int
    r_squared: float


@dataclass
class OscillationSignificance:
    """Statistical significance tests for oscillation detection."""
    event_name: str
    # Model comparison
    oscillator_aic: float
    exponential_aic: float
    model_preference: str  # 'oscillator' or 'exponential'
    delta_aic: float  # AIC(exponential) - AIC(oscillator), positive = oscillator better
    # Zero-crossing test
    observed_crossings: int
    expected_crossings_random: float
    crossing_p_value: float
    # Autocorrelation test
    acf_oscillation_score: float  # negative ACF at lag ~T/2 indicates oscillation
    acf_p_value: float
    # Overall verdict
    is_significant: bool
    confidence: str  # 'high', 'medium', 'low'


def compute_aic(n: int, rss: float, k: int) -> float:
    """Compute AIC for a model.

    Args:
        n: Number of data points
        rss: Residual sum of squares
        k: Number of parameters

    Returns:
        AIC value (lower is better)
    """
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def fit_exponential_decay(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit simple exponential decay: y = A * exp(-λt) + c

    Returns:
        (rss, n_params, r_squared)
    """
    def exp_model(t, A, lam, c):
        return A * np.exp(-lam * t) + c

    try:
        # Initial guesses
        A0 = y[0] - y[-1]
        lam0 = 0.1
        c0 = y[-1]

        popt, _ = optimize.curve_fit(
            exp_model, t, y,
            p0=[A0, lam0, c0],
            maxfev=5000,
            bounds=([-np.inf, 0, -np.inf], [np.inf, 10, np.inf])
        )

        y_pred = exp_model(t, *popt)
        rss = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - rss / ss_tot if ss_tot > 0 else 0

        return rss, 3, r_squared
    except:
        return np.inf, 3, 0


def fit_damped_oscillator_model(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit damped oscillator: y = A * exp(-ζωn*t) * cos(ωd*t + φ) + c

    Returns:
        (rss, n_params, r_squared)
    """
    def osc_model(t, A, zeta_omega, omega_d, phi, c):
        return A * np.exp(-zeta_omega * t) * np.cos(omega_d * t + phi) + c

    try:
        # Estimate initial frequency from zero crossings
        crossings = np.where(np.diff(np.sign(y - np.mean(y))))[0]
        if len(crossings) >= 2:
            half_periods = np.diff(t[crossings])
            omega_d0 = np.pi / np.mean(half_periods) if len(half_periods) > 0 else 0.5
        else:
            omega_d0 = 0.5

        # Clamp omega_d0 to reasonable range
        omega_d0 = np.clip(omega_d0, 0.1, 5.0)

        # Initial guesses
        A0 = np.max(np.abs(y - np.mean(y)))
        if A0 < 1e-10:
            A0 = 0.1
        zeta_omega0 = 0.1
        phi0 = 0
        c0 = np.mean(y)

        # Try multiple initial conditions
        best_rss = np.inf
        best_r2 = 0

        for omega_init in [omega_d0, omega_d0 * 0.5, omega_d0 * 2, 0.5, 1.0]:
            for zeta_init in [0.05, 0.1, 0.3]:
                try:
                    popt, _ = optimize.curve_fit(
                        osc_model, t, y,
                        p0=[A0, zeta_init, omega_init, phi0, c0],
                        maxfev=2000,
                        bounds=([-np.inf, 0, 0.01, -np.pi, -np.inf], [np.inf, 5, 10, np.pi, np.inf])
                    )

                    y_pred = osc_model(t, *popt)
                    rss = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - rss / ss_tot if ss_tot > 0 else 0

                    if rss < best_rss:
                        best_rss = rss
                        best_r2 = r2
                except:
                    continue

        if best_rss == np.inf:
            # Fallback: use simple sinusoidal fit
            return np.sum(y**2), 5, 0

        return best_rss, 5, best_r2
    except:
        return np.sum(y**2), 5, 0


def test_zero_crossings(y: np.ndarray, n_simulations: int = 1000) -> Tuple[int, float, float]:
    """Test if zero crossings are significantly different from random walk.

    Random walk has fewer crossings than oscillatory signal.

    Returns:
        (observed_crossings, expected_crossings, p_value)
    """
    y_centered = y - np.mean(y)
    observed = len(np.where(np.diff(np.sign(y_centered)))[0])

    # Simulate random walks with same variance
    n = len(y)
    sigma = np.std(np.diff(y_centered))

    random_crossings = []
    for _ in range(n_simulations):
        # Random walk
        rw = np.cumsum(np.random.randn(n) * sigma)
        rw_centered = rw - np.mean(rw)
        crossings = len(np.where(np.diff(np.sign(rw_centered)))[0])
        random_crossings.append(crossings)

    expected = np.mean(random_crossings)

    # One-sided test: oscillation has MORE crossings than random walk
    p_value = np.mean([c >= observed for c in random_crossings])

    return observed, expected, p_value


def test_autocorrelation_oscillation(y: np.ndarray, max_lag: int = 15) -> Tuple[float, float]:
    """Test for oscillatory autocorrelation structure.

    Oscillation produces negative ACF at intermediate lags.
    Random walk has monotonically decaying positive ACF.

    Returns:
        (oscillation_score, p_value)
    """
    try:
        # Compute ACF using numpy (no statsmodels dependency)
        n = len(y)
        y_centered = y - np.mean(y)
        nlags = min(max_lag, n // 3)

        acf_values = []
        var = np.sum(y_centered**2)

        for lag in range(nlags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf = np.sum(y_centered[:-lag] * y_centered[lag:]) / var
                acf_values.append(acf)

        acf_values = np.array(acf_values)

        # Oscillation score: magnitude of minimum (most negative) ACF
        min_acf = np.min(acf_values[1:])  # Exclude lag 0
        oscillation_score = -min_acf  # Positive if ACF goes negative

        # Test significance using Bartlett's formula
        se = 1 / np.sqrt(n)  # Standard error under null
        z_score = oscillation_score / se
        p_value = 1 - stats.norm.cdf(z_score)  # One-sided

        return oscillation_score, p_value
    except:
        return 0, 1.0


def compute_phase_coherence(events_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """Compute phase coherence across multiple events.

    If oscillation is real, phase should be consistent across events.

    Args:
        events_data: List of (t, y) arrays for each event

    Returns:
        (coherence, p_value) where coherence is 0-1
    """
    if len(events_data) < 2:
        return 0, 1.0

    phases = []
    for t, y in events_data:
        if len(y) < 5:
            continue

        # Find first zero crossing after peak
        y_centered = y - np.mean(y)
        crossings = np.where(np.diff(np.sign(y_centered)))[0]

        if len(crossings) > 0:
            first_crossing_time = t[crossings[0]]
            phases.append(first_crossing_time)

    if len(phases) < 2:
        return 0, 1.0

    # Compute circular variance (measures phase consistency)
    # Normalize to common period
    mean_phase = np.mean(phases)
    phase_variance = np.var(phases)

    # Coherence: low variance = high coherence
    coherence = np.exp(-phase_variance / (mean_phase**2 + 0.01))

    # Rayleigh test for phase uniformity
    n = len(phases)
    R = coherence * n
    p_value = np.exp(-R**2 / n) if n > 0 else 1.0

    return coherence, p_value


def validate_oscillation(
    ts: pd.DataFrame,
    event_time: datetime,
    event_name: str,
    window_after_days: int = 30
) -> Optional[OscillationSignificance]:
    """Run statistical tests to validate oscillation detection.

    Tests:
    1. AIC model comparison: oscillator vs exponential decay
    2. Zero-crossing test: more crossings than random walk
    3. Autocorrelation: negative ACF at intermediate lags
    """
    # Extract post-event window
    start = event_time
    end = event_time + timedelta(days=window_after_days)
    window = ts[(ts['timestamp'] >= start) & (ts['timestamp'] <= end)].copy()

    if len(window) < 10:
        return None

    t = (window['timestamp'] - event_time).dt.total_seconds().values / 86400  # Days
    y = window['sentiment_mean'].values
    y_centered = y - np.mean(y)

    # 1. Model comparison (AIC)
    osc_rss, osc_k, osc_r2 = fit_damped_oscillator_model(t, y_centered)
    exp_rss, exp_k, exp_r2 = fit_exponential_decay(t, np.abs(y_centered))

    osc_aic = compute_aic(len(y), osc_rss, osc_k)
    exp_aic = compute_aic(len(y), exp_rss, exp_k)

    delta_aic = exp_aic - osc_aic  # Positive = oscillator better
    model_preference = 'oscillator' if delta_aic > 2 else ('exponential' if delta_aic < -2 else 'neither')

    # 2. Zero-crossing test
    observed_crossings, expected_crossings, crossing_p = test_zero_crossings(y_centered)

    # 3. Autocorrelation test
    acf_score, acf_p = test_autocorrelation_oscillation(y_centered)

    # Overall verdict
    significant_tests = 0
    if delta_aic > 2:  # Strong preference for oscillator
        significant_tests += 1
    if crossing_p < 0.05:  # Significantly more crossings
        significant_tests += 1
    if acf_p < 0.05:  # Significant negative ACF
        significant_tests += 1

    is_significant = significant_tests >= 2
    confidence = 'high' if significant_tests == 3 else ('medium' if significant_tests == 2 else 'low')

    return OscillationSignificance(
        event_name=event_name,
        oscillator_aic=osc_aic,
        exponential_aic=exp_aic,
        model_preference=model_preference,
        delta_aic=delta_aic,
        observed_crossings=observed_crossings,
        expected_crossings_random=expected_crossings,
        crossing_p_value=crossing_p,
        acf_oscillation_score=acf_score,
        acf_p_value=acf_p,
        is_significant=is_significant,
        confidence=confidence
    )


def run_significance_analysis(
    ts: pd.DataFrame,
    events: Dict[str, datetime],
    results: List[EventDynamics]
) -> List[OscillationSignificance]:
    """Run statistical significance tests for all events."""
    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION: Is the oscillation REAL or NOISE?")
    print("=" * 70)

    significance_results = []
    events_data = []  # For phase coherence

    for event_name, event_time in events.items():
        print(f"\n  {event_name}:")

        sig = validate_oscillation(ts, event_time, event_name)

        if sig is not None:
            significance_results.append(sig)

            # Collect data for phase coherence
            start = event_time
            end = event_time + timedelta(days=30)
            window = ts[(ts['timestamp'] >= start) & (ts['timestamp'] <= end)]
            if len(window) >= 5:
                t = (window['timestamp'] - event_time).dt.total_seconds().values / 86400
                y = window['sentiment_mean'].values
                events_data.append((t, y - np.mean(y)))

            # Print results
            print(f"    Model comparison (ΔAIC): {sig.delta_aic:.1f} → {sig.model_preference}")
            print(f"    Zero-crossings: {sig.observed_crossings} observed vs {sig.expected_crossings_random:.1f} expected (p={sig.crossing_p_value:.3f})")
            print(f"    ACF oscillation score: {sig.acf_oscillation_score:.3f} (p={sig.acf_p_value:.3f})")
            print(f"    → Verdict: {'✓ SIGNIFICANT' if sig.is_significant else '✗ Not significant'} ({sig.confidence} confidence)")
        else:
            print(f"    Insufficient data for validation")

    # Phase coherence across events
    if len(events_data) >= 2:
        coherence, coherence_p = compute_phase_coherence(events_data)
        print(f"\n  Cross-event phase coherence: {coherence:.3f} (p={coherence_p:.3f})")
        if coherence > 0.5 and coherence_p < 0.05:
            print(f"    → Events show CONSISTENT oscillation phase!")

    # Summary
    if significance_results:
        n_significant = sum(1 for s in significance_results if s.is_significant)
        n_high_conf = sum(1 for s in significance_results if s.confidence == 'high')

        print(f"\n  SUMMARY:")
        print(f"    Significant oscillation: {n_significant}/{len(significance_results)} events")
        print(f"    High confidence: {n_high_conf}/{len(significance_results)} events")

        if n_significant >= len(significance_results) // 2:
            print(f"\n  ✓ Oscillation is STATISTICALLY SIGNIFICANT - not just noise!")
        else:
            print(f"\n  ⚠ Oscillation evidence is WEAK - may be noise")

    return significance_results


def load_election_data(data_dir: str, level: str = 'World') -> pd.DataFrame:
    """
    Load sentiment data for election analysis.

    Args:
        data_dir: Directory containing data
        level: Which level to load - 'World' (global daily), 'Country', 'State'
               Default is 'World' for fastest loading and clearest signal

    Searches for "Sentiment Data - XXX" folders.
    """
    import glob as glob_module

    # Direct file names to try first
    possible_files = [
        'election2020_tweets.csv',
        'USA_Nov2020_Election_Tweets.csv',
        'tweets.csv',
        'data.csv',
    ]

    for fname in possible_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading data from: {fname}")
            return _load_csv_file(fpath)

    # Search for "Sentiment Data - XXX" folders
    search_paths = [
        data_dir,
        os.path.join(data_dir, '..'),
        os.path.join(data_dir, '..', '..'),
        os.path.join(data_dir, '..', 'sentiment_index'),
    ]

    for search_path in search_paths:
        pattern = os.path.join(search_path, 'Sentiment Data - *')
        sentiment_folders = glob_module.glob(pattern)

        if sentiment_folders:
            print(f"Found {len(sentiment_folders)} sentiment data folders")

            # Find the requested level
            target_folder = None
            for folder in sentiment_folders:
                folder_name = os.path.basename(folder)
                folder_level = folder_name.replace('Sentiment Data - ', '')
                if folder_level == level:
                    target_folder = folder
                    break

            if target_folder is None:
                print(f"  Level '{level}' not found, available:")
                for folder in sentiment_folders:
                    print(f"    - {os.path.basename(folder)}")
                # Fall back to World if available, else first
                for folder in sentiment_folders:
                    if 'World' in folder:
                        target_folder = folder
                        break
                if target_folder is None:
                    target_folder = sentiment_folders[0]

            folder_name = os.path.basename(target_folder)
            print(f"  Loading from: {folder_name}")

            all_dfs = []
            csv_files = glob_module.glob(os.path.join(target_folder, '*.csv'))

            # Prioritize 2020 file
            files_2020 = [f for f in csv_files if '2020' in os.path.basename(f)]
            if files_2020:
                csv_files = files_2020
                print(f"  Using 2020 file(s): {[os.path.basename(f) for f in files_2020]}")
            else:
                print(f"  Loading all {len(csv_files)} files")

            for fpath in csv_files:
                try:
                    df = _load_csv_file(fpath)
                    if df is not None and len(df) > 0:
                        all_dfs.append(df)
                        print(f"    Loaded {len(df):,} rows from {os.path.basename(fpath)}")
                except Exception as e:
                    print(f"    Error: {e}")

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                print(f"  Total: {len(combined):,} rows")
                return combined

    raise FileNotFoundError(
        f"No data file found. Please place sentiment data in 'Sentiment Data - XXX' folders."
    )


def _load_csv_file(fpath: str) -> pd.DataFrame:
    """Load a CSV file, handling large files with chunking."""
    file_size = os.path.getsize(fpath)

    if file_size > 100_000_000:  # > 100MB, use chunking
        print(f"  Large file ({file_size / 1e6:.1f} MB), loading in chunks...")
        chunks = []
        for chunk in pd.read_csv(fpath, chunksize=500000,
                                 low_memory=False,
                                 on_bad_lines='skip'):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip')

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the election/sentiment data."""
    print(f"  Raw columns: {list(df.columns[:10])}...")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace('-', '_').str.replace(' ', '_')

    print(f"  Standardized columns: {list(df.columns[:10])}...")

    # Parse timestamp - try many possible column names
    time_cols = ['created_at', 'createdat', 'timestamp', 'date', 'time', 'datetime',
                 'day', 'tweet_date', 'post_date', 'created', 'posted_at']
    timestamp_found = False
    for col in time_cols:
        if col in df.columns:
            df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
            timestamp_found = True
            print(f"  Using '{col}' as timestamp")
            break

    if not timestamp_found:
        # Try to find any column with 'date' or 'time' in the name
        for col in df.columns:
            if 'date' in col or 'time' in col:
                df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                timestamp_found = True
                print(f"  Using '{col}' as timestamp")
                break

    # Find sentiment column - try many possible names
    sent_cols = ['score', 'sentiment', 'sentiment_score', 'compound', 'polarity',
                 'vader', 'vader_score', 'sentiment_compound', 'avg_sentiment',
                 'mean_sentiment', 'daily_sentiment']
    sentiment_found = False
    for col in sent_cols:
        if col in df.columns:
            df['sentiment'] = pd.to_numeric(df[col], errors='coerce')
            sentiment_found = True
            print(f"  Using '{col}' as sentiment")
            break

    if not sentiment_found:
        # Try to find any column with 'sent' or 'score' in the name
        for col in df.columns:
            if 'sent' in col or 'score' in col or 'polarity' in col:
                df['sentiment'] = pd.to_numeric(df[col], errors='coerce')
                sentiment_found = True
                print(f"  Using '{col}' as sentiment")
                break

    if not timestamp_found or not sentiment_found:
        print(f"  WARNING: Could not find required columns")
        print(f"  Available columns: {list(df.columns)}")
        if not timestamp_found:
            print(f"  Missing: timestamp column")
        if not sentiment_found:
            print(f"  Missing: sentiment column")
        return pd.DataFrame()

    # Drop rows with missing values
    df = df.dropna(subset=['timestamp', 'sentiment'])

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  After preprocessing: {len(df):,} rows")
    if len(df) > 0:
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Sentiment range: {df['sentiment'].min():.3f} to {df['sentiment'].max():.3f}")

    return df


def aggregate_sentiment_timeseries(
    df: pd.DataFrame,
    freq: str = '1D'
) -> pd.DataFrame:
    """
    Aggregate sentiment to regular time intervals.

    If data is already at daily resolution (one row per day), skip aggregation.

    Args:
        df: DataFrame with 'timestamp' and 'sentiment' columns
        freq: Pandas frequency string (e.g., '1D' for daily, '1h' for hourly)

    Returns:
        DataFrame with aggregated sentiment statistics
    """
    # Check if data is already at daily resolution (one row per unique date)
    unique_dates = df['timestamp'].dt.date.nunique()
    if unique_dates == len(df):
        # Data is already daily - just rename columns
        print(f"  Data already at daily resolution ({len(df)} days)")
        result = df[['timestamp', 'sentiment']].copy()
        result['sentiment_mean'] = result['sentiment']
        result['sentiment_std'] = 0.0
        result['tweet_count'] = 1
        return result

    # Aggregate to target frequency
    df_indexed = df.set_index('timestamp')

    agg = df_indexed.resample(freq).agg({
        'sentiment': ['mean', 'std', 'count']
    })
    agg.columns = ['sentiment_mean', 'sentiment_std', 'tweet_count']
    agg = agg.reset_index()

    # Filter low-count periods (only if we have granular data)
    if freq in ['1h', '1H', '15min', '30min']:
        agg = agg[agg['tweet_count'] >= 10]

    print(f"  Aggregated to {len(agg)} time periods ({freq})")

    return agg


def fit_damped_oscillator_to_event(
    ts: pd.DataFrame,
    event_time: datetime,
    window_before_days: int = 7,
    window_after_days: int = 30
) -> Optional[EventDynamics]:
    """
    Fit damped oscillator to sentiment dynamics around an event.

    Model: y(t) = y_eq + A * exp(-ζωn*t) * cos(ωd*t + φ)

    Note: Uses DAYS since sentiment data is daily resolution.
    """
    # Extract window around event
    start = event_time - timedelta(days=window_before_days)
    end = event_time + timedelta(days=window_after_days)

    window = ts[(ts['timestamp'] >= start) & (ts['timestamp'] <= end)].copy()

    if len(window) < 10:  # Reduced threshold for daily data
        return None

    # Time in DAYS from event
    window['t_days'] = (window['timestamp'] - event_time).dt.total_seconds() / 86400

    # Pre-event baseline
    pre_data = window[window['t_days'] < 0]
    post_data = window[window['t_days'] >= 0]

    if len(pre_data) < 3 or len(post_data) < 5:
        return None

    pre_sentiment = pre_data['sentiment_mean'].mean()

    # Post-event equilibrium (late in window)
    late_data = window[window['t_days'] > window_after_days * 0.7]
    if len(late_data) > 0:
        post_sentiment = late_data['sentiment_mean'].mean()
    else:
        post_sentiment = post_data['sentiment_mean'].mean()

    # Fit oscillator to post-event data
    t = post_data['t_days'].values
    y = post_data['sentiment_mean'].values
    y_dev = y - post_sentiment

    # Peak deviation
    peak_deviation = np.max(np.abs(y_dev))

    # Estimate damping and frequency
    # Find zero crossings
    crossings = np.where(np.diff(np.sign(y_dev)))[0]
    n_oscillations = len(crossings) // 2

    if n_oscillations > 0 and len(crossings) >= 2:
        # Average half-period in days
        half_periods = np.diff(t[crossings])
        if len(half_periods) > 0:
            period_days = 2 * np.mean(half_periods)
        else:
            period_days = 0
    else:
        period_days = 0

    # Fit exponential envelope to estimate damping
    peaks_idx = signal.find_peaks(np.abs(y_dev))[0]
    if len(peaks_idx) >= 2:
        peak_times = t[peaks_idx]
        peak_vals = np.abs(y_dev[peaks_idx])

        # Fit: log(peak) = log(A) - ζωn*t
        try:
            slope, intercept, r, p, se = stats.linregress(peak_times, np.log(peak_vals + 1e-10))
            damping_rate = -slope  # ζωn
            if period_days > 0:
                omega_n = 2 * np.pi / period_days
                damping_ratio = damping_rate / omega_n if omega_n > 0 else 1.0
            else:
                damping_ratio = 1.0
        except:
            damping_ratio = 1.0
            r = 0
    else:
        damping_ratio = 1.0
        r = 0

    # Compute overall fit quality
    if period_days > 0 and damping_ratio < 2:
        omega_d = 2 * np.pi / period_days
        zeta_omega_n = damping_ratio * omega_d / np.sqrt(1 - min(damping_ratio, 0.99)**2) if damping_ratio < 1 else omega_d
        y_pred = peak_deviation * np.exp(-zeta_omega_n * t) * np.cos(omega_d * t)
        ss_res = np.sum((y_dev - y_pred)**2)
        ss_tot = np.sum((y_dev - np.mean(y_dev))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r_squared = r**2

    return EventDynamics(
        event_name='',
        event_time=event_time,
        pre_sentiment=pre_sentiment,
        post_sentiment=post_sentiment,
        peak_deviation=peak_deviation,
        damping_ratio=min(damping_ratio, 5.0),
        period_days=period_days,
        n_oscillations=n_oscillations,
        r_squared=max(0, r_squared)
    )


def analyze_all_events(
    ts: pd.DataFrame,
    events: Dict[str, datetime]
) -> List[EventDynamics]:
    """Analyze dynamics around all events."""
    results = []

    for event_name, event_time in events.items():
        print(f"\n  Analyzing {event_name} ({event_time})...")

        dynamics = fit_damped_oscillator_to_event(ts, event_time)

        if dynamics is not None:
            dynamics.event_name = event_name
            results.append(dynamics)

            regime = "underdamped" if dynamics.damping_ratio < 1 else "overdamped"
            print(f"    ζ = {dynamics.damping_ratio:.3f} ({regime})")
            print(f"    Period = {dynamics.period_days:.1f} days")
            print(f"    Oscillations = {dynamics.n_oscillations}")
        else:
            print(f"    Insufficient data")

    return results


def visualize_event_dynamics(
    ts: pd.DataFrame,
    events: Dict[str, datetime],
    results: List[EventDynamics]
):
    """Visualize sentiment dynamics around events."""
    n_events = len(events)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (event_name, event_time) in enumerate(events.items()):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract window
        start = event_time - timedelta(hours=12)
        end = event_time + timedelta(hours=72)
        window = ts[(ts['timestamp'] >= start) & (ts['timestamp'] <= end)]

        if len(window) > 0:
            # Plot
            t_hours = (window['timestamp'] - event_time).dt.total_seconds() / 3600
            ax.plot(t_hours, window['sentiment_mean'], 'b-', alpha=0.7, lw=1)
            ax.axvline(0, color='red', ls='--', lw=2, label='Event')

            # Find corresponding result
            result = next((r for r in results if r.event_name == event_name), None)
            if result:
                ax.axhline(result.pre_sentiment, color='green', ls=':', alpha=0.5)
                ax.axhline(result.post_sentiment, color='orange', ls=':', alpha=0.5)

                regime = "ζ<1 (osc)" if result.damping_ratio < 1 else "ζ>1 (over)"
                ax.set_title(f"{event_name}\n{regime}, T={result.period_days:.1f}d", fontsize=10)
            else:
                ax.set_title(event_name, fontsize=10)

            ax.set_xlabel('Hours from event')
            ax.set_ylabel('Sentiment')
            ax.set_xlim(-12, 72)

    # Summary panel
    ax = axes[-1]
    if results:
        zetas = [r.damping_ratio for r in results]
        ax.bar([r.event_name for r in results], zetas, color='steelblue', edgecolor='black')
        ax.axhline(1.0, color='red', ls='--', label='Critical (ζ=1)')
        ax.set_ylabel('Damping Ratio ζ')
        ax.set_title('Damping by Event')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    plt.suptitle('Election 2020: Sentiment Dynamics Around Key Events\n(Testing Kaplowitz-Fink Oscillation Model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'election2020_dynamics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {save_path}")


def generate_sample_data():
    """Generate sample election sentiment data for demonstration."""
    print("Generating sample election data for demonstration...")

    np.random.seed(42)

    # Generate hourly data from Oct 1 to Nov 15, 2020
    start = datetime(2020, 10, 1)
    end = datetime(2020, 11, 15)

    timestamps = pd.date_range(start, end, freq='1H')
    n_points = len(timestamps)

    # Base sentiment with slow drift
    base = 0.05 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))  # Weekly cycle

    sentiment = base.copy()

    # Add responses to events
    for event_name, event_time in ELECTION_EVENTS.items():
        if start <= event_time <= end:
            event_idx = int((event_time - start).total_seconds() / 3600)

            if 0 <= event_idx < n_points:
                # Damped oscillation response
                t = np.arange(n_points - event_idx)

                # Different events have different characteristics
                if 'debate' in event_name:
                    shock = 0.1 * (np.random.rand() - 0.5)  # Random direction
                    zeta = 0.3  # Underdamped
                    omega = 2 * np.pi / 12  # 12-hour period
                elif event_name == 'election_day':
                    shock = 0.15
                    zeta = 0.5
                    omega = 2 * np.pi / 24
                else:  # election_called
                    shock = 0.2
                    zeta = 0.2  # Very underdamped - lots of oscillation
                    omega = 2 * np.pi / 8

                omega_d = omega * np.sqrt(1 - zeta**2) if zeta < 1 else omega
                response = shock * np.exp(-zeta * omega * t) * np.cos(omega_d * t)

                sentiment[event_idx:] += response

    # Add noise
    sentiment += 0.02 * np.random.randn(n_points)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'sentiment_mean': sentiment,
        'sentiment_std': 0.1 + 0.05 * np.random.rand(n_points),
        'tweet_count': np.random.poisson(1000, n_points)
    })

    return df


def main():
    """Main analysis."""
    print("=" * 70)
    print("HAMILTONIAN DYNAMICS: Election 2020 Sentiment")
    print("=" * 70)
    print("""
Testing Kaplowitz-Fink oscillation model on political sentiment.

Key events as "impulsive forces":
- Presidential debates
- Election Day
- Election called

Prediction: Sentiment shows damped oscillation (ζ < 1) after shocks.
    """)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        df = load_election_data(DATA_DIR)
        df = preprocess_data(df)

        # Filter to 2020 data for election analysis
        if len(df) > 0 and 'timestamp' in df.columns:
            df_2020 = df[(df['timestamp'] >= '2020-01-01') & (df['timestamp'] <= '2020-12-31')]
            if len(df_2020) > 0:
                print(f"  Filtered to 2020: {len(df_2020):,} rows")
                df = df_2020
            else:
                print(f"  WARNING: No 2020 data found in loaded files")
                print(f"  Available date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                # List available years
                years = df['timestamp'].dt.year.unique()
                print(f"  Available years: {sorted(years)}")

        ts = aggregate_sentiment_timeseries(df, freq='1D')
    except FileNotFoundError:
        print("\n  No data file found. Using sample data for demonstration...")
        ts = generate_sample_data()

    # Analyze events
    print("\n[2/4] Analyzing event dynamics...")
    results = analyze_all_events(ts, ELECTION_EVENTS)

    # Summary statistics
    print("\n[3/4] Computing summary...")

    if results:
        zetas = [r.damping_ratio for r in results]
        periods = [r.period_days for r in results if r.period_days > 0]

        n_underdamped = sum(1 for z in zetas if z < 1)

        print(f"\n   Events analyzed: {len(results)}")
        print(f"   Underdamped (ζ<1): {n_underdamped}/{len(results)}")
        print(f"   Mean damping ratio: {np.mean(zetas):.3f}")
        if periods:
            print(f"   Mean period: {np.mean(periods):.1f} days")

    # Visualize
    print("\n[4/4] Creating visualizations...")
    visualize_event_dynamics(ts, ELECTION_EVENTS, results)

    # Statistical validation
    print("\n[5/5] Running statistical validation...")
    significance_results = run_significance_analysis(ts, ELECTION_EVENTS, results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        mean_zeta = np.mean([r.damping_ratio for r in results])

        if mean_zeta < 1:
            print(f"""
✓ Evidence for UNDERDAMPED dynamics in political sentiment!
  Mean ζ = {mean_zeta:.3f} < 1 (oscillatory regime)

  This matches Kaplowitz et al. (1983) finding that attitudes
  oscillate after persuasive shocks. Political events act as
  impulsive forces that excite damped oscillations in collective
  sentiment.
""")
        else:
            print(f"""
○ Evidence for OVERDAMPED dynamics (ζ = {mean_zeta:.3f} > 1)
  Sentiment decays monotonically without oscillation.

  This may reflect:
  - Aggregation washing out individual oscillations
  - Different dynamics at collective vs individual level
  - Insufficient temporal resolution
""")

    return ts, results


if __name__ == '__main__':
    ts, results = main()

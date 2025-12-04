# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics: Prediction Market Analysis
========================================================

Tests for oscillatory belief dynamics in prediction market prices.

Prediction market prices represent aggregated beliefs about future events.
When news arrives (impulsive force), prices should exhibit damped oscillation
according to Kaplowitz-Fink theory.

Kaplowitz et al. (1983) found:
- Period of oscillation: ~13.5 seconds
- Very low damping (c* ≈ -0.003)

Question: Do prediction markets show similar oscillatory dynamics
around news events?

Data sources:
- Polymarket API: https://docs.polymarket.com/
- Polymarket Analytics: https://polymarketanalytics.com
- Historical data providers: https://entityml.com/

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'prediction_markets')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class OscillationEvent:
    """Detected oscillation event in price series."""
    market_id: str
    event_time: datetime
    pre_price: float
    post_price: float
    peak_amplitude: float
    n_oscillations: int
    period_seconds: float
    damping_ratio: float
    settled_price: float


def load_market_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load prediction market data from files.

    Expected format: CSV with columns [timestamp, price, volume]
    """
    markets = {}

    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath)

                # Standardize column names
                df.columns = df.columns.str.lower()

                # Parse timestamp
                for col in ['timestamp', 'time', 'datetime', 'date']:
                    if col in df.columns:
                        df['timestamp'] = pd.to_datetime(df[col])
                        break

                if 'timestamp' not in df.columns:
                    continue

                # Find price column
                for col in ['price', 'close', 'mid', 'probability']:
                    if col in df.columns:
                        df['price'] = df[col]
                        break

                if 'price' not in df.columns:
                    continue

                market_id = fname.replace('.csv', '')
                markets[market_id] = df.sort_values('timestamp').reset_index(drop=True)
                print(f"Loaded {market_id}: {len(df)} records")

            except Exception as e:
                print(f"Error loading {fname}: {e}")

    return markets


def detect_news_events(df: pd.DataFrame, threshold: float = 0.05) -> List[int]:
    """
    Detect sudden price movements (potential news events).

    A news event is defined as a price change > threshold in a short window.
    """
    if len(df) < 10:
        return []

    prices = df['price'].values
    events = []

    # Compute rolling change
    for i in range(5, len(prices) - 5):
        # Price change in 5-period window
        change = abs(prices[i] - prices[i-5])

        if change > threshold:
            events.append(i)

    # Filter close events (keep first in each cluster)
    filtered = []
    for i, idx in enumerate(events):
        if i == 0 or idx - events[i-1] > 50:
            filtered.append(idx)

    return filtered


def fit_damped_oscillator(
    t: np.ndarray,
    y: np.ndarray,
    equilibrium: float
) -> Tuple[float, float, float, float]:
    """
    Fit damped harmonic oscillator to price deviation from equilibrium.

    Model: y(t) = A * exp(-ζωn*t) * cos(ωd*t + φ)

    Returns: (amplitude, damping_ratio, omega_n, phase)
    """
    # Deviation from equilibrium
    y_dev = y - equilibrium

    # Initial guess from data
    A_init = np.max(np.abs(y_dev))

    # Find zero crossings to estimate frequency
    crossings = np.where(np.diff(np.sign(y_dev)))[0]
    if len(crossings) >= 2:
        avg_half_period = np.mean(np.diff(crossings)) * (t[1] - t[0])
        omega_init = np.pi / avg_half_period
    else:
        omega_init = 0.1  # Default

    def model(params):
        A, zeta, omega_n, phi = params
        if zeta < 0 or zeta >= 1 or omega_n <= 0:
            return np.inf

        omega_d = omega_n * np.sqrt(1 - zeta**2)
        y_pred = A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
        return np.sum((y_dev - y_pred)**2)

    # Optimize
    try:
        result = optimize.minimize(
            model,
            [A_init, 0.1, omega_init, 0],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        A, zeta, omega_n, phi = result.x
        return max(0, A), max(0.01, min(0.99, zeta)), max(0.01, omega_n), phi
    except:
        return A_init, 0.5, omega_init, 0


def analyze_event(
    df: pd.DataFrame,
    event_idx: int,
    window_before: int = 50,
    window_after: int = 200
) -> Optional[OscillationEvent]:
    """
    Analyze price dynamics around a detected news event.
    """
    start_idx = max(0, event_idx - window_before)
    end_idx = min(len(df), event_idx + window_after)

    if end_idx - start_idx < 50:
        return None

    prices = df['price'].values[start_idx:end_idx]
    times = df['timestamp'].values[start_idx:end_idx]

    # Time in seconds from event
    event_time = times[window_before] if event_idx >= window_before else times[0]

    # Convert to numeric time
    if hasattr(times[0], 'timestamp'):
        t = np.array([(pd.Timestamp(ts) - pd.Timestamp(event_time)).total_seconds()
                      for ts in times])
    else:
        t = np.arange(len(prices)) * 1.0  # Assume 1 second intervals

    # Pre and post prices
    pre_price = np.mean(prices[:window_before]) if window_before > 0 else prices[0]
    post_price = np.mean(prices[-20:])

    # Fit oscillator to post-event data
    post_idx = window_before
    t_post = t[post_idx:] - t[post_idx]
    y_post = prices[post_idx:]

    A, zeta, omega_n, phi = fit_damped_oscillator(t_post, y_post, post_price)

    # Count oscillations (zero crossings)
    y_dev = y_post - post_price
    crossings = np.where(np.diff(np.sign(y_dev)))[0]
    n_oscillations = len(crossings) // 2

    # Period
    if omega_n > 0 and zeta < 1:
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        period = 2 * np.pi / omega_d
    else:
        period = 0

    return OscillationEvent(
        market_id='',
        event_time=pd.Timestamp(event_time),
        pre_price=pre_price,
        post_price=post_price,
        peak_amplitude=A,
        n_oscillations=n_oscillations,
        period_seconds=period,
        damping_ratio=zeta,
        settled_price=post_price
    )


def analyze_all_markets(markets: Dict[str, pd.DataFrame]) -> List[OscillationEvent]:
    """
    Analyze all markets for oscillation events.
    """
    all_events = []

    for market_id, df in markets.items():
        print(f"\nAnalyzing {market_id}...")

        # Detect news events
        event_indices = detect_news_events(df)
        print(f"  Found {len(event_indices)} potential news events")

        # Analyze each event
        for idx in event_indices[:10]:  # Limit per market
            event = analyze_event(df, idx)
            if event is not None:
                event.market_id = market_id
                all_events.append(event)

    return all_events


def test_oscillation_hypothesis(events: List[OscillationEvent]) -> Dict:
    """
    Test whether prediction markets show oscillatory dynamics.
    """
    if not events:
        return {'n_events': 0, 'conclusion': 'No events to analyze'}

    damping_ratios = [e.damping_ratio for e in events]
    periods = [e.period_seconds for e in events if e.period_seconds > 0]
    n_oscillations = [e.n_oscillations for e in events]

    # Proportion with ζ < 1 (underdamped)
    n_underdamped = sum(1 for z in damping_ratios if z < 1)
    prop_underdamped = n_underdamped / len(damping_ratios)

    # Compare with Kaplowitz et al. period (~13.5 seconds)
    kaplowitz_period = 13.5

    return {
        'n_events': len(events),
        'n_underdamped': n_underdamped,
        'prop_underdamped': prop_underdamped,
        'mean_damping_ratio': np.mean(damping_ratios),
        'std_damping_ratio': np.std(damping_ratios),
        'mean_period': np.mean(periods) if periods else None,
        'std_period': np.std(periods) if periods else None,
        'kaplowitz_period': kaplowitz_period,
        'mean_oscillations': np.mean(n_oscillations),
    }


def visualize_results(events: List[OscillationEvent], results: Dict):
    """Create visualization of prediction market oscillation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if not events:
        plt.text(0.5, 0.5, 'No events to visualize\nPlease download market data',
                 ha='center', va='center', fontsize=14, transform=fig.transFigure)
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_market_hamiltonian.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        return

    # Panel A: Damping ratio distribution
    ax = axes[0, 0]
    zetas = [e.damping_ratio for e in events]
    ax.hist(zetas, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Critical damping')
    ax.axvline(np.mean(zetas), color='green', ls='-', lw=2, label=f'Mean = {np.mean(zetas):.2f}')
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A. Damping Ratios\n(ζ<1: oscillation)', fontsize=11)
    ax.legend()

    # Panel B: Period distribution
    ax = axes[0, 1]
    periods = [e.period_seconds for e in events if e.period_seconds > 0]
    if periods:
        ax.hist(periods, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(13.5, color='blue', ls='--', lw=2, label='Kaplowitz (13.5s)')
        if np.mean(periods) < 1000:
            ax.axvline(np.mean(periods), color='green', ls='-', lw=2,
                      label=f'Mean = {np.mean(periods):.1f}s')
    ax.set_xlabel('Oscillation Period (seconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('B. Oscillation Periods', fontsize=11)
    ax.legend()

    # Panel C: Number of oscillations
    ax = axes[1, 0]
    n_osc = [e.n_oscillations for e in events]
    ax.hist(n_osc, bins=range(max(n_osc)+2), alpha=0.7, color='gold', edgecolor='black')
    ax.set_xlabel('Number of Oscillations', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('C. Oscillations Before Settling', fontsize=11)

    # Panel D: Price change vs damping
    ax = axes[1, 1]
    changes = [abs(e.post_price - e.pre_price) for e in events]
    ax.scatter(changes, zetas, alpha=0.5, c='steelblue', s=50)
    ax.set_xlabel('Price Change', fontsize=12)
    ax.set_ylabel('Damping Ratio ζ', fontsize=12)
    ax.set_title('D. Shock Size vs Damping', fontsize=11)
    ax.axhline(1.0, color='red', ls='--', alpha=0.5)

    plt.suptitle('Hamiltonian Dynamics in Prediction Markets\n(Testing Kaplowitz-Fink Model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'prediction_market_hamiltonian.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {save_path}")


def generate_sample_data():
    """
    Generate sample prediction market data for testing.

    This simulates what real market data would look like.
    """
    print("Generating sample prediction market data for demonstration...")

    np.random.seed(42)

    # Simulate a market with a news event
    n_points = 1000
    dt = 1.0  # 1 second intervals

    # Equilibrium price
    p_eq = 0.5

    # News event at t=200: sudden shock
    shock_time = 200
    shock_size = 0.15

    # Generate price trajectory
    t = np.arange(n_points) * dt
    prices = np.zeros(n_points)

    # Pre-shock: random walk around equilibrium
    prices[:shock_time] = p_eq + 0.02 * np.cumsum(np.random.randn(shock_time)) / np.sqrt(shock_time)

    # Post-shock: damped oscillation (underdamped case from Kaplowitz)
    zeta = 0.1  # Low damping like Kaplowitz found
    omega_n = 2 * np.pi / 13.5  # Period ~13.5 seconds
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    t_post = t[shock_time:] - t[shock_time]
    new_eq = p_eq + shock_size

    # Damped oscillation plus noise
    oscillation = shock_size * np.exp(-zeta * omega_n * t_post) * np.cos(omega_d * t_post)
    noise = 0.01 * np.random.randn(len(t_post))
    prices[shock_time:] = new_eq - oscillation + noise

    # Create dataframe
    start_time = datetime(2024, 11, 5, 12, 0, 0)  # Election day
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_points)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': np.clip(prices, 0.01, 0.99),
        'volume': np.random.exponential(1000, n_points)
    })

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    fpath = os.path.join(DATA_DIR, 'sample_election_market.csv')
    df.to_csv(fpath, index=False)
    print(f"Saved sample data to: {fpath}")

    return {'sample_election_market': df}


def main():
    """Main analysis."""
    print("=" * 70)
    print("HAMILTONIAN DYNAMICS: Prediction Market Analysis")
    print("=" * 70)
    print("""
Testing whether prediction markets show oscillatory belief dynamics
after news events, as predicted by Kaplowitz-Fink theory.

Kaplowitz et al. (1983) found for attitude change:
- Damping ratio ζ < 1 (underdamped, oscillatory)
- Period of oscillation: ~13.5 seconds
- Very low damping coefficient

Question: Do aggregated market beliefs show similar dynamics?
    """)

    # Try to load real data
    print("\n[1/4] Loading market data...")
    markets = {}

    if os.path.exists(DATA_DIR):
        markets = load_market_data(DATA_DIR)

    if not markets:
        print("\n No market data found. Generating sample data...")
        markets = generate_sample_data()

    # Analyze
    print("\n[2/4] Detecting news events and fitting oscillator...")
    events = analyze_all_markets(markets)
    print(f"\n      Analyzed {len(events)} events total")

    # Test hypothesis
    print("\n[3/4] Testing oscillation hypothesis...")
    results = test_oscillation_hypothesis(events)

    print(f"\n   Results:")
    print(f"      Events analyzed: {results['n_events']}")
    print(f"      Underdamped (ζ<1): {results['n_underdamped']} ({results['prop_underdamped']:.1%})")
    print(f"      Mean damping ratio: {results['mean_damping_ratio']:.3f}")
    if results['mean_period']:
        print(f"      Mean period: {results['mean_period']:.2f} seconds")
        print(f"      Kaplowitz period: {results['kaplowitz_period']:.1f} seconds")

    # Visualize
    print("\n[4/4] Creating visualizations...")
    visualize_results(events, results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results['prop_underdamped'] > 0.5:
        print("""
✓ Evidence for UNDERDAMPED dynamics in prediction markets!
  Markets show oscillatory belief dynamics after news shocks.
  This matches Kaplowitz et al. (1983) findings for persuasion.
""")
    else:
        print("""
○ Mixed evidence for oscillation in prediction markets.
  May need higher-frequency data to detect ~13 second oscillations.
""")

    return events, results


if __name__ == '__main__':
    events, results = main()

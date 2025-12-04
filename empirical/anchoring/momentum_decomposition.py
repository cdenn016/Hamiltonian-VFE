# -*- coding: utf-8 -*-
"""
Momentum Decomposition: Environmental vs Behavioral
====================================================

Is the observed momentum (r = 0.405) trivially explained by the
environment, or is there a behavioral/cognitive component?

Key test: Compare observed momentum to expected momentum from
independent Bernoulli draws with the same true probabilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'belief_updates', 'data')


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'sampledata.csv'))


def compute_expected_momentum(true_prob):
    """
    For independent Bernoulli(p) draws, compute expected correlation
    between consecutive updates.

    Update = +1 (green) with prob p, -1 (red) with prob 1-p

    E[update] = p(+1) + (1-p)(-1) = 2p - 1
    E[update²] = p(1) + (1-p)(1) = 1
    Var[update] = 1 - (2p-1)² = 4p(1-p)

    For INDEPENDENT draws:
    Cov(update_t, update_{t+1}) = 0

    So expected correlation = 0 for independent samples!

    But wait - updates are DIFFERENCES in cumulative evidence:
    update_t = evidence_t - evidence_{t-1}

    If samples are independent Bernoulli:
    sample_t ~ Bernoulli(p), coded as +1 or -1
    evidence_t = sum of samples up to t
    update_t = sample_t (just the new sample!)

    So update_t and update_{t+1} are just consecutive samples,
    which ARE independent. Expected correlation = 0.
    """
    # For independent Bernoulli, expected correlation = 0
    return 0.0


def simulate_null_momentum(true_probs, n_samples_per_trial, n_simulations=100):
    """
    Simulate momentum under null hypothesis of independent sampling.

    For each trial, generate independent Bernoulli samples and compute
    the momentum correlation.
    """
    simulated_rs = []

    for sim in range(n_simulations):
        all_updates_t = []
        all_updates_t1 = []

        for i, (p, n) in enumerate(zip(true_probs, n_samples_per_trial)):
            if n < 4:
                continue

            # Generate independent Bernoulli samples
            samples = np.random.choice([1, -1], size=int(n),
                                       p=[p, 1-p])

            # Updates are just the samples (change in evidence)
            updates = samples[:-1]  # update_t
            updates_next = samples[1:]  # update_{t+1}

            all_updates_t.extend(updates[:-1])
            all_updates_t1.extend(updates_next[:-1])

        if len(all_updates_t) > 100:
            r, _ = stats.pearsonr(all_updates_t, all_updates_t1)
            simulated_rs.append(r)

    return np.array(simulated_rs)


def extract_real_momentum_data(df):
    """Extract actual momentum data from the experiment."""
    momentum_data = []
    trial_info = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 5:
            continue

        green = sampling['green'].values
        red = sampling['red'].values
        evidence = green - red
        updates = np.diff(evidence)

        true_prob = group.iloc[0]['recip']
        n_samples = len(sampling)

        trial_info.append({
            'true_prob': true_prob,
            'n_samples': n_samples
        })

        for i in range(len(updates) - 1):
            momentum_data.append({
                'update_t': updates[i],
                'update_t1': updates[i + 1],
                'true_prob': true_prob,
            })

    return pd.DataFrame(momentum_data), pd.DataFrame(trial_info)


def compute_momentum_by_prob(mom_df):
    """Compute momentum separately for each true probability."""
    results = []

    for prob in sorted(mom_df['true_prob'].unique()):
        subset = mom_df[mom_df['true_prob'] == prob]
        if len(subset) > 100:
            r, p = stats.pearsonr(subset['update_t'], subset['update_t1'])
            results.append({
                'true_prob': prob,
                'observed_r': r,
                'n': len(subset),
                'expected_r': 0.0,  # Independent Bernoulli
            })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("MOMENTUM DECOMPOSITION: Is it Environmental or Behavioral?")
    print("=" * 70)

    print("\n[1] Loading data...")
    df = load_data()

    print("[2] Extracting momentum data...")
    mom_df, trial_info = extract_real_momentum_data(df)
    print(f"    {len(mom_df):,} update pairs from {len(trial_info):,} trials")

    # Observed momentum
    observed_r, observed_p = stats.pearsonr(mom_df['update_t'], mom_df['update_t1'])
    print(f"\n[3] OBSERVED MOMENTUM: r = {observed_r:.4f}")

    # Theoretical expected momentum (independent Bernoulli)
    print(f"\n[4] THEORETICAL EXPECTED (independent samples): r = 0.0000")

    # Simulate null distribution
    print("\n[5] Simulating null distribution (100 simulations)...")
    null_rs = simulate_null_momentum(
        trial_info['true_prob'].values,
        trial_info['n_samples'].values,
        n_simulations=100
    )

    print(f"    Null distribution: mean = {null_rs.mean():.4f}, std = {null_rs.std():.4f}")
    print(f"    95% CI: [{np.percentile(null_rs, 2.5):.4f}, {np.percentile(null_rs, 97.5):.4f}]")

    # Z-score
    z = (observed_r - null_rs.mean()) / null_rs.std()
    print(f"\n[6] Z-SCORE: {z:.1f}")
    print(f"    Observed r is {z:.0f} standard deviations above null")

    # Momentum by true probability
    print("\n[7] Momentum by true probability:")
    by_prob = compute_momentum_by_prob(mom_df)
    print(by_prob.to_string(index=False))

    # Key insight
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if z > 10:
        print("""
The observed momentum (r = {:.3f}) is VASTLY higher than expected
from independent Bernoulli sampling (r ≈ 0).

This means the momentum is NOT just environmental autocorrelation.
There's a genuine behavioral/cognitive component.

Possible explanations:
1. Confirmation bias: people sample in streaks
2. Attention fluctuations: sustained focus on one direction
3. Motor/perceptual inertia: tendency to repeat actions
4. Belief momentum: once moving in a direction, continue

This supports a Hamiltonian component in belief dynamics,
even though τ doesn't scale with Λ.
""".format(observed_r))
    else:
        print("""
The observed momentum can be largely explained by the environmental
structure. No strong evidence for behavioral momentum.
""")

    return mom_df, null_rs, by_prob


if __name__ == '__main__':
    mom_df, null_rs, by_prob = main()

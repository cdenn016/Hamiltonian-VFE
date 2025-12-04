# -*- coding: utf-8 -*-
"""
Belief Inertia Analysis: Social Trust Dataset
==============================================

Testing Hamiltonian VFE prediction: τ ∝ Λ (precision → inertia)

Dataset: Ma, Westhoff & van Duijvenvoorde (2022)
"Uncertainty about others' trustworthiness increases during adolescence"

Key question: Does accumulated evidence (precision) predict resistance to
updating, or does it speed up decision-making?

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'belief_updates', 'data')


def load_trust_data():
    """Load the social trust sampling data."""
    data_path = os.path.join(DATA_DIR, 'sampledata.csv')
    df = pd.read_csv(data_path)
    return df


def extract_trial_summaries(df):
    """
    Extract summary statistics for each trial.

    For each trial, we get:
    - n_samples: total samples before decision
    - final_evidence: green - red at decision
    - true_prob: actual trustworthiness (recip)
    - decision: trust (1) or distrust (-1)
    - evidence_for_decision: was final evidence aligned with decision?
    """
    trials = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        # Get final row (the decision)
        final_row = group[group['choice'] != 0]
        if len(final_row) == 0:
            continue
        final_row = final_row.iloc[-1]

        n_samples = final_row['green'] + final_row['red']
        final_evidence = final_row['green'] - final_row['red']
        true_prob = final_row['recip']
        decision = final_row['choice']
        age = final_row['age']

        # Track sampling trajectory
        sampling_rows = group[group['choice'] == 0]
        if len(sampling_rows) > 1:
            evidence_trajectory = (sampling_rows['green'] - sampling_rows['red']).values
        else:
            evidence_trajectory = []

        trials.append({
            'subject': subj,
            'trial': trial,
            'age': age,
            'n_samples': n_samples,
            'final_evidence': final_evidence,
            'true_prob': true_prob,
            'decision': decision,
            'evidence_trajectory': evidence_trajectory,
            'n_trajectory_points': len(evidence_trajectory),
        })

    return pd.DataFrame(trials)


def test_inertia_in_sampling(trial_df):
    """
    Test: Does accumulated evidence predict MORE sampling (inertia)?

    Hamiltonian prediction: Higher precision (more samples) should mean
    more resistance to changing, which could manifest as:
    - More samples needed when prior is strong
    - Slower switching when evidence reverses
    """
    print("=" * 70)
    print("TEST 1: Does accumulated precision predict sampling behavior?")
    print("=" * 70)

    # Group by true probability (high vs low trustworthiness)
    high_trust = trial_df[trial_df['true_prob'] >= 0.6]
    low_trust = trial_df[trial_df['true_prob'] <= 0.4]

    print(f"\nHigh trustworthiness trials (p >= 0.6): n = {len(high_trust)}")
    print(f"  Mean samples before decision: {high_trust['n_samples'].mean():.2f}")
    print(f"Low trustworthiness trials (p <= 0.4): n = {len(low_trust)}")
    print(f"  Mean samples before decision: {low_trust['n_samples'].mean():.2f}")

    # Test: Do people who sample MORE have stronger final evidence?
    # (This would suggest precision → confidence → decision)
    r, p = stats.spearmanr(trial_df['n_samples'], trial_df['final_evidence'].abs())
    print(f"\nCorrelation: n_samples vs |final_evidence|")
    print(f"  r = {r:.3f}, p = {p:.4f}")

    if r > 0 and p < 0.05:
        print("  → More sampling leads to stronger evidence (expected)")
    elif r < 0 and p < 0.05:
        print("  → More sampling with WEAKER evidence (inertia/difficulty)")

    return r, p


def test_evidence_reversal_inertia(df):
    """
    Test: When evidence reverses direction, does prior strength predict
    how long it takes to switch?

    This directly tests τ ∝ Λ: stronger prior → longer adjustment time
    """
    print("\n" + "=" * 70)
    print("TEST 2: Evidence reversal and adjustment time")
    print("=" * 70)

    reversal_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 4:
            continue

        evidence = (sampling['green'] - sampling['red']).values

        # Find sign changes in evidence
        signs = np.sign(evidence)
        sign_changes = np.where(np.diff(signs) != 0)[0]

        for change_idx in sign_changes:
            if change_idx < 2 or change_idx > len(evidence) - 3:
                continue

            # Evidence BEFORE reversal (strength of prior)
            evidence_before = evidence[:change_idx + 1]
            prior_strength = np.abs(evidence_before[-1])  # |evidence at reversal|
            n_samples_before = change_idx + 1  # Precision proxy

            # How long to reach opposite sign after reversal?
            evidence_after = evidence[change_idx + 1:]
            original_sign = signs[change_idx]

            # Find when evidence reaches opposite direction
            opposite_indices = np.where(np.sign(evidence_after) == -original_sign)[0]
            if len(opposite_indices) > 0:
                samples_to_reverse = opposite_indices[0] + 1
            else:
                samples_to_reverse = len(evidence_after)  # Didn't fully reverse

            reversal_data.append({
                'subject': subj,
                'trial': trial,
                'prior_strength': prior_strength,
                'n_samples_before': n_samples_before,
                'samples_to_reverse': samples_to_reverse,
                'precision_proxy': n_samples_before,  # More samples = higher precision
            })

    if len(reversal_data) < 20:
        print(f"  Only {len(reversal_data)} reversals found - insufficient data")
        return None, None

    rev_df = pd.DataFrame(reversal_data)
    print(f"\nFound {len(rev_df)} evidence reversal events")

    # KEY TEST: Does prior precision predict reversal time?
    r_prec, p_prec = stats.spearmanr(rev_df['precision_proxy'],
                                      rev_df['samples_to_reverse'])
    r_strength, p_strength = stats.spearmanr(rev_df['prior_strength'],
                                              rev_df['samples_to_reverse'])

    print(f"\nτ (samples to reverse) vs precision proxies:")
    print(f"  τ ~ n_samples_before: r = {r_prec:.3f} (p = {p_prec:.4f})")
    print(f"  τ ~ |evidence_before|: r = {r_strength:.3f} (p = {p_strength:.4f})")

    if r_prec > 0 and p_prec < 0.05:
        print("\n  ✓ SUPPORTS INERTIA: Higher precision → slower reversal")
    elif r_prec < 0 and p_prec < 0.05:
        print("\n  ✗ CONTRADICTS INERTIA: Higher precision → FASTER reversal")
    else:
        print("\n  ○ No significant relationship")

    return rev_df, (r_prec, p_prec, r_strength, p_strength)


def test_age_modulation(trial_df, rev_df=None):
    """
    Test: Does age modulate inertia effects?

    Paper finding: Adolescents have MORE uncertainty in priors,
    so they should show LESS inertia (faster updating).
    """
    print("\n" + "=" * 70)
    print("TEST 3: Age modulation of belief updating")
    print("=" * 70)

    # Bin by age groups
    trial_df = trial_df.copy()
    trial_df['age_group'] = pd.cut(trial_df['age'],
                                    bins=[9, 13, 17, 25],
                                    labels=['10-13', '14-17', '18-24'])

    print("\nMean samples before decision by age:")
    for age_group, group in trial_df.groupby('age_group'):
        print(f"  {age_group}: {group['n_samples'].mean():.2f} samples")

    # Correlation: age vs n_samples
    r_age, p_age = stats.spearmanr(trial_df['age'], trial_df['n_samples'])
    print(f"\nAge vs n_samples: r = {r_age:.3f} (p = {p_age:.4f})")

    if r_age > 0 and p_age < 0.05:
        print("  → Older = more sampling (more cautious/precise)")
    elif r_age < 0 and p_age < 0.05:
        print("  → Younger = more sampling (more uncertain)")

    return r_age, p_age


def test_bayesian_vs_inertia(trial_df):
    """
    Compare: Bayesian updating vs inertia model

    Bayesian: Decision based on likelihood ratio (green/red evidence)
    Inertia: Decision influenced by prior + resistance to change
    """
    print("\n" + "=" * 70)
    print("TEST 4: Bayesian vs Inertia model comparison")
    print("=" * 70)

    # For each trial, compute:
    # - Bayesian prediction: trust if green > red
    # - Whether decision matches Bayesian prediction

    trial_df = trial_df.copy()
    trial_df['bayesian_pred'] = np.sign(trial_df['final_evidence'])
    trial_df['bayesian_pred'] = trial_df['bayesian_pred'].replace(0, 1)  # Tie → trust

    # Match rate
    matches = (trial_df['bayesian_pred'] == trial_df['decision'])
    trial_df['bayesian_match'] = matches

    match_rate = matches.mean()
    print(f"\nBayesian prediction accuracy: {match_rate:.1%}")

    # Does evidence strength predict accuracy?
    # Strong evidence → should match Bayesian more
    evidence_strength = trial_df['final_evidence'].abs()
    r_strength, p_strength = stats.spearmanr(evidence_strength, matches.astype(int))
    print(f"Evidence strength vs Bayesian match: r = {r_strength:.3f} (p = {p_strength:.4f})")

    # KEY: Does n_samples predict DEVIATION from Bayesian?
    # If inertia: more samples → more deviation (stuck on prior)
    # If no inertia: more samples → more accuracy
    r_samples, p_samples = stats.spearmanr(trial_df['n_samples'],
                                            matches.astype(int))
    print(f"n_samples vs Bayesian match: r = {r_samples:.3f} (p = {p_samples:.4f})")

    if r_samples > 0 and p_samples < 0.05:
        print("  → More sampling = MORE Bayesian (no inertia)")
    elif r_samples < 0 and p_samples < 0.05:
        print("  → More sampling = LESS Bayesian (inertia effect)")

    return match_rate, (r_samples, p_samples)


def analyze_conservatism(trial_df):
    """
    Test Edwards-style conservatism: Do people under-update?

    Compare actual decisions to Bayesian optimal given evidence.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Edwards-style conservatism analysis")
    print("=" * 70)

    # For varying evidence levels, what's the trust rate?
    evidence_bins = pd.cut(trial_df['final_evidence'],
                           bins=[-10, -3, -1, 1, 3, 10],
                           labels=['strong_neg', 'weak_neg', 'neutral',
                                   'weak_pos', 'strong_pos'])

    print("\nTrust rate by evidence strength:")
    for label in ['strong_neg', 'weak_neg', 'neutral', 'weak_pos', 'strong_pos']:
        mask = evidence_bins == label
        if mask.sum() > 0:
            trust_rate = (trial_df.loc[mask, 'decision'] == 1).mean()
            print(f"  {label}: {trust_rate:.1%} trust ({mask.sum()} trials)")

    # Conservatism metric: Slope of trust rate vs evidence
    # Bayesian should be steep sigmoid, conservative is flatter
    evidence = trial_df['final_evidence'].values
    trust = (trial_df['decision'] == 1).astype(float).values

    # Simple slope
    slope, intercept = np.polyfit(evidence, trust, 1)
    print(f"\nSlope (trust rate vs evidence): {slope:.3f}")
    print("  (Lower slope = more conservative/inertia)")

    return slope


def run_analysis():
    """Run full inertia analysis."""
    print("=" * 70)
    print("BELIEF INERTIA ANALYSIS: SOCIAL TRUST TASK")
    print("=" * 70)
    print("\nTesting Hamiltonian prediction: τ ∝ Λ (precision → inertia)")
    print("If true: More accumulated evidence → slower belief adjustment")

    # Load data
    print("\n[1/6] Loading data...")
    df = load_trust_data()
    print(f"  Loaded {len(df)} observations")
    print(f"  {df['subject'].nunique()} subjects, ages {df['age'].min()}-{df['age'].max()}")

    # Extract trial summaries
    print("\n[2/6] Extracting trial summaries...")
    trial_df = extract_trial_summaries(df)
    print(f"  {len(trial_df)} complete trials")

    # Test 1: Sampling behavior
    print("\n[3/6] Testing sampling behavior...")
    test_inertia_in_sampling(trial_df)

    # Test 2: Evidence reversal
    print("\n[4/6] Testing evidence reversal dynamics...")
    rev_df, reversal_stats = test_evidence_reversal_inertia(df)

    # Test 3: Age modulation
    print("\n[5/6] Testing age modulation...")
    test_age_modulation(trial_df, rev_df)

    # Test 4: Bayesian vs Inertia
    print("\n[6/6] Comparing Bayesian vs Inertia models...")
    test_bayesian_vs_inertia(trial_df)

    # Test 5: Conservatism
    analyze_conservatism(trial_df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if reversal_stats is not None:
        r_prec, p_prec = reversal_stats[0], reversal_stats[1]
        if r_prec > 0 and p_prec < 0.05:
            print("\n✓ Evidence for INERTIA: Precision predicts slower reversal")
        elif r_prec < 0 and p_prec < 0.05:
            print("\n✗ Evidence AGAINST inertia: Precision predicts FASTER reversal")
            print("  (Same pattern as Nassar data - surprise-driven updating)")
        else:
            print("\n○ Inconclusive: No significant precision-inertia relationship")

    return trial_df, rev_df


if __name__ == '__main__':
    trial_df, rev_df = run_analysis()

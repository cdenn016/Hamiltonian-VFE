# -*- coding: utf-8 -*-
"""
Initial Commitment Analysis: When Does Inertia Emerge?
=======================================================

The previous analysis found:
- Evidence reversal: precision → FASTER adjustment (anti-inertia)
- BUT: more sampling → less Bayesian (slight inertia)

This suggests inertia might emerge specifically when people form
an INITIAL COMMITMENT and then resist changing it.

Key hypothesis: Inertia appears not in response to evidence,
but in COMMITMENT to an initial interpretation.

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_trust_data():
    """Load the social trust sampling data."""
    df = pd.read_csv('/home/user/Hamiltonian-VFE/empirical/belief_updates/data/sampledata.csv')
    return df


def analyze_initial_commitment(df):
    """
    Test: Does early evidence create a "commitment" that resists later reversal?

    If inertia exists, people who see STRONG early evidence in one direction
    should be slower to switch even when later evidence reverses.
    """
    print("=" * 70)
    print("INITIAL COMMITMENT ANALYSIS")
    print("=" * 70)
    print("\nHypothesis: Early evidence creates 'commitment' that resists reversal")

    commitment_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 6:
            continue

        green = sampling['green'].values
        red = sampling['red'].values
        evidence = green - red

        # Early evidence (first 3 samples)
        early_evidence = evidence[2] if len(evidence) > 2 else evidence[-1]
        early_direction = np.sign(early_evidence)

        # Find if evidence ever reversed from early direction
        if early_direction == 0:
            continue

        # Track when evidence goes opposite to early direction
        opposite_indices = np.where(np.sign(evidence) == -early_direction)[0]

        if len(opposite_indices) == 0:
            # Never reversed - consistent evidence
            reversal_occurred = False
            samples_after_reversal = 0
            final_matches_early = True
        else:
            reversal_occurred = True
            reversal_point = opposite_indices[0]
            samples_after_reversal = len(evidence) - reversal_point

            # Did final decision match early direction or follow the reversal?
            final_row = group[group['choice'] != 0]
            if len(final_row) > 0:
                final_decision = final_row.iloc[-1]['choice']
                final_matches_early = (final_decision == early_direction) or \
                                     (early_direction > 0 and final_decision == 1) or \
                                     (early_direction < 0 and final_decision == -1)
            else:
                continue

        # Final evidence at decision
        final_row = group[group['choice'] != 0]
        if len(final_row) == 0:
            continue
        final_evidence = final_row.iloc[-1]['green'] - final_row.iloc[-1]['red']

        # Strength of early commitment
        early_strength = abs(early_evidence)

        commitment_data.append({
            'subject': subj,
            'trial': trial,
            'early_direction': early_direction,
            'early_strength': early_strength,
            'reversal_occurred': reversal_occurred,
            'samples_after_reversal': samples_after_reversal,
            'final_matches_early': final_matches_early,
            'final_evidence': final_evidence,
            'total_samples': len(evidence),
            'age': group.iloc[0]['age'],
        })

    commit_df = pd.DataFrame(commitment_data)
    print(f"\nAnalyzed {len(commit_df)} trials")

    # Analysis 1: Does early strength predict sticking with early direction?
    print("\n" + "-" * 40)
    print("TEST 1: Early commitment strength vs final alignment")
    print("-" * 40)

    # Only look at trials where reversal occurred
    reversed_trials = commit_df[commit_df['reversal_occurred']]
    print(f"\nTrials with evidence reversal: {len(reversed_trials)}")

    if len(reversed_trials) > 50:
        # Does early strength predict sticking with early direction?
        r_commit, p_commit = stats.spearmanr(
            reversed_trials['early_strength'],
            reversed_trials['final_matches_early'].astype(int)
        )
        print(f"\nEarly strength vs final matches early:")
        print(f"  r = {r_commit:.3f} (p = {p_commit:.4f})")

        if r_commit > 0 and p_commit < 0.05:
            print("  ✓ INERTIA: Stronger early evidence → more likely to stick with it")
        elif r_commit < 0 and p_commit < 0.05:
            print("  ✗ Anti-inertia: Stronger early evidence → more likely to switch")

        # Match rate by early strength bins
        reversed_trials = reversed_trials.copy()
        reversed_trials['early_bin'] = pd.cut(
            reversed_trials['early_strength'],
            bins=[0, 1, 2, 10],
            labels=['weak(1)', 'medium(2)', 'strong(3+)']
        )

        print("\nStick-with-early rate by early evidence strength:")
        for label in ['weak(1)', 'medium(2)', 'strong(3+)']:
            mask = reversed_trials['early_bin'] == label
            if mask.sum() > 10:
                rate = reversed_trials.loc[mask, 'final_matches_early'].mean()
                print(f"  {label}: {rate:.1%} stick with early ({mask.sum()} trials)")

    # Analysis 2: Does early commitment override final evidence?
    print("\n" + "-" * 40)
    print("TEST 2: Does early commitment override final evidence?")
    print("-" * 40)

    # Look at trials where final evidence DISAGREES with early direction
    # Do people still follow early direction?
    conflicting = reversed_trials[
        np.sign(reversed_trials['final_evidence']) != reversed_trials['early_direction']
    ]
    print(f"\nTrials where final evidence opposes early direction: {len(conflicting)}")

    if len(conflicting) > 50:
        # What fraction still match early?
        stick_rate = conflicting['final_matches_early'].mean()
        print(f"  Fraction sticking with early direction: {stick_rate:.1%}")

        if stick_rate > 0.5:
            print("  ✓ INERTIA: Majority stick with early despite contrary evidence")

            # Does early strength amplify this?
            r_override, p_override = stats.spearmanr(
                conflicting['early_strength'],
                conflicting['final_matches_early'].astype(int)
            )
            print(f"\n  Early strength vs override rate:")
            print(f"  r = {r_override:.3f} (p = {p_override:.4f})")
        else:
            print("  ✗ No inertia: Majority follow final evidence")

    # Analysis 3: Age differences in commitment
    print("\n" + "-" * 40)
    print("TEST 3: Age differences in initial commitment")
    print("-" * 40)

    reversed_trials = reversed_trials.copy()
    reversed_trials['age_group'] = pd.cut(
        reversed_trials['age'],
        bins=[9, 15, 25],
        labels=['young(10-15)', 'older(16-24)']
    )

    for age_group in ['young(10-15)', 'older(16-24)']:
        mask = reversed_trials['age_group'] == age_group
        if mask.sum() > 50:
            rate = reversed_trials.loc[mask, 'final_matches_early'].mean()
            print(f"  {age_group}: {rate:.1%} stick with early ({mask.sum()} trials)")

    # Correlation
    r_age, p_age = stats.spearmanr(
        reversed_trials['age'],
        reversed_trials['final_matches_early'].astype(int)
    )
    print(f"\nAge vs stick-with-early: r = {r_age:.3f} (p = {p_age:.4f})")

    return commit_df, reversed_trials


def analyze_sampling_as_inertia(df):
    """
    Alternative interpretation: Sampling MORE is itself a form of inertia.

    People who are uncertain keep sampling rather than committing.
    This "action inertia" (keep doing what you're doing) might be the
    manifestation of belief inertia.
    """
    print("\n" + "=" * 70)
    print("SAMPLING AS ACTION INERTIA")
    print("=" * 70)

    trial_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0]
        final = group[group['choice'] != 0]

        if len(final) == 0:
            continue

        n_samples = len(sampling)
        final_row = final.iloc[-1]
        final_evidence = final_row['green'] - final_row['red']
        decision = final_row['choice']
        true_prob = final_row['recip']

        # Was decision optimal given true probability?
        optimal = (true_prob >= 0.5 and decision == 1) or \
                  (true_prob < 0.5 and decision == -1)

        trial_data.append({
            'subject': subj,
            'trial': trial,
            'n_samples': n_samples,
            'final_evidence': final_evidence,
            'decision': decision,
            'true_prob': true_prob,
            'optimal': optimal,
            'age': group.iloc[0]['age'],
        })

    trial_df = pd.DataFrame(trial_data)

    # Key test: Does sampling MORE predict worse decisions?
    print("\nHypothesis: Over-sampling is action inertia (avoiding commitment)")

    r_samples_optimal, p = stats.spearmanr(trial_df['n_samples'], trial_df['optimal'].astype(int))
    print(f"\nn_samples vs optimal decision: r = {r_samples_optimal:.3f} (p = {p:.4f})")

    # Bin by sampling
    trial_df['sample_bin'] = pd.cut(
        trial_df['n_samples'],
        bins=[0, 5, 10, 20, 100],
        labels=['quick(1-5)', 'moderate(6-10)', 'slow(11-20)', 'very_slow(21+)']
    )

    print("\nOptimal decision rate by sampling duration:")
    for label in ['quick(1-5)', 'moderate(6-10)', 'slow(11-20)', 'very_slow(21+)']:
        mask = trial_df['sample_bin'] == label
        if mask.sum() > 50:
            rate = trial_df.loc[mask, 'optimal'].mean()
            print(f"  {label}: {rate:.1%} optimal ({mask.sum()} trials)")

    # Control for evidence strength
    print("\n" + "-" * 40)
    print("Controlling for evidence strength:")
    print("-" * 40)

    # Strong evidence trials only
    strong_evidence = trial_df[trial_df['final_evidence'].abs() >= 3]
    if len(strong_evidence) > 100:
        r_strong, p_strong = stats.spearmanr(
            strong_evidence['n_samples'],
            strong_evidence['optimal'].astype(int)
        )
        print(f"\nWith strong evidence (|e| >= 3):")
        print(f"  n_samples vs optimal: r = {r_strong:.3f} (p = {p_strong:.4f})")
        print(f"  n = {len(strong_evidence)} trials")

    return trial_df


def analyze_belief_momentum(df):
    """
    Test: Do updates show momentum (current update predicts next update)?

    In Hamiltonian dynamics, momentum means dμ/dt at time t predicts dμ/dt at t+1.
    Here we test: does update[t] predict update[t+1]?
    """
    print("\n" + "=" * 70)
    print("BELIEF MOMENTUM ANALYSIS")
    print("=" * 70)

    momentum_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 5:
            continue

        # Compute "updates" as changes in evidence interpretation
        green = sampling['green'].values
        red = sampling['red'].values
        evidence = green - red

        # Update = change in evidence
        updates = np.diff(evidence)

        if len(updates) < 3:
            continue

        # Test: does update[t] predict update[t+1]?
        for i in range(len(updates) - 1):
            momentum_data.append({
                'subject': subj,
                'trial': trial,
                'update_t': updates[i],
                'update_t1': updates[i + 1],
                'evidence_t': evidence[i],
                'sample_num': i,
                'age': group.iloc[0]['age'],
            })

    mom_df = pd.DataFrame(momentum_data)
    print(f"\nAnalyzed {len(mom_df)} update pairs")

    # Test momentum: update[t] → update[t+1]
    r_momentum, p_momentum = stats.pearsonr(mom_df['update_t'], mom_df['update_t1'])
    print(f"\nMomentum test: update[t] vs update[t+1]")
    print(f"  r = {r_momentum:.3f} (p = {p_momentum:.4f})")

    if r_momentum > 0 and p_momentum < 0.05:
        print("  ✓ POSITIVE MOMENTUM: Updates persist")
    elif r_momentum < 0 and p_momentum < 0.05:
        print("  ✗ NEGATIVE MOMENTUM: Updates reverse (oscillation)")
    else:
        print("  ○ No significant momentum")

    # Conditional on evidence sign
    print("\n" + "-" * 40)
    print("Momentum by evidence direction:")
    print("-" * 40)

    positive_ev = mom_df[mom_df['evidence_t'] > 0]
    negative_ev = mom_df[mom_df['evidence_t'] < 0]

    for label, subset in [('positive evidence', positive_ev), ('negative evidence', negative_ev)]:
        if len(subset) > 100:
            r, p = stats.pearsonr(subset['update_t'], subset['update_t1'])
            print(f"  {label}: r = {r:.3f} (p = {p:.4f}, n = {len(subset)})")

    return mom_df


def run_analysis():
    """Run full initial commitment analysis."""
    print("=" * 70)
    print("WHEN DOES INERTIA EMERGE?")
    print("=" * 70)
    print("\nPrevious finding: Precision → FASTER reversal (anti-inertia)")
    print("But: More sampling → less Bayesian (slight inertia)")
    print("\nThis analysis explores WHERE inertia might hide...")

    df = load_trust_data()

    # Test 1: Initial commitment
    print("\n")
    commit_df, reversed_df = analyze_initial_commitment(df)

    # Test 2: Sampling as inertia
    print("\n")
    trial_df = analyze_sampling_as_inertia(df)

    # Test 3: Belief momentum
    print("\n")
    mom_df = analyze_belief_momentum(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Where Does Inertia Emerge?")
    print("=" * 70)

    return commit_df, trial_df, mom_df


if __name__ == '__main__':
    results = run_analysis()

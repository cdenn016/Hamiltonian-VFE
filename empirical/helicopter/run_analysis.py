#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Complete Inertia of Belief Analysis
========================================

Main script for testing Hamiltonian-VFE predictions against
Nassar helicopter task data.

Usage:
    python -m empirical.helicopter.run_analysis
    python -m empirical.helicopter.run_analysis --subjects 1 2 3
    python -m empirical.helicopter.run_analysis --quick

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Handle both module execution and direct script execution
try:
    from .data_loader import (
        load_mcguire_nassar_2014,
        compute_summary_stats
    )
    from .fitting import (
        compare_dynamics,
        summarize_comparisons
    )
    from .analysis import (
        analyze_all_subjects,
        summarize_inertia_evidence
    )
    from .visualization import (
        plot_belief_trajectory,
        plot_changepoint_response,
        plot_model_comparison_summary,
        plot_inertia_evidence_summary
    )
except ImportError:
    # Running as script - add parent directories to path
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from empirical.helicopter.data_loader import (
        load_mcguire_nassar_2014,
        compute_summary_stats
    )
    from empirical.helicopter.fitting import (
        compare_dynamics,
        summarize_comparisons
    )
    from empirical.helicopter.analysis import (
        analyze_all_subjects,
        summarize_inertia_evidence
    )
    from empirical.helicopter.visualization import (
        plot_belief_trajectory,
        plot_changepoint_response,
        plot_model_comparison_summary,
        plot_inertia_evidence_summary
    )


def run_full_analysis(subject_ids: list = None,
                      output_dir: Path = None,
                      quick: bool = False):
    """
    Run complete analysis pipeline.

    Args:
        subject_ids: List of subject IDs to analyze (None = all)
        output_dir: Directory for output files
        quick: If True, skip optimization for faster results
    """
    print("="*60)
    print("INERTIA OF BELIEF - EMPIRICAL ANALYSIS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup output directory
    if output_dir is None:
        output_dir = Path('_results/helicopter_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n[1/5] Loading data...")
    all_subjects = load_mcguire_nassar_2014()

    if subject_ids:
        subjects = {sid: all_subjects[sid] for sid in subject_ids if sid in all_subjects}
    else:
        subjects = all_subjects

    print(f"  Loaded {len(subjects)} subjects")

    # Data summary
    data_summary = compute_summary_stats(subjects)
    print(f"  Total trials: {data_summary['total_trials']}")
    print(f"  Mean learning rate: {data_summary['mean_learning_rate']:.3f}")

    # Fit models
    print("\n[2/5] Fitting Gradient vs Hamiltonian models...")
    comparisons = compare_dynamics(subjects, optimize=not quick, verbose=True)

    # Model comparison summary
    print("\n[3/5] Summarizing model comparison...")
    model_summary = summarize_comparisons(comparisons)

    print("\n  MODEL COMPARISON RESULTS:")
    print(f"  ─────────────────────────")
    print(f"  Hamiltonian wins: {model_summary['n_hamiltonian_wins']}/{model_summary['n_subjects']} "
          f"({100*model_summary['hamiltonian_win_rate']:.1f}%)")
    print(f"  Mean MSE improvement: {100*model_summary['mean_mse_improvement']:.1f}%")
    print(f"  Gradient MSE: {model_summary['gradient_mean_mse']:.2f} ± {model_summary['gradient_std_mse']:.2f}")
    print(f"  Hamiltonian MSE: {model_summary['hamiltonian_mean_mse']:.2f} ± {model_summary['hamiltonian_std_mse']:.2f}")

    print(f"\n  FITTED PARAMETERS:")
    print(f"  ─────────────────────────")
    print(f"  Gradient LR: {model_summary['gradient_lr_mean']:.3f} ± {model_summary['gradient_lr_std']:.3f}")
    print(f"  Hamiltonian mass: {model_summary['hamiltonian_mass_mean']:.2f} ± {model_summary['hamiltonian_mass_std']:.2f}")
    print(f"  Hamiltonian friction: {model_summary['hamiltonian_friction_mean']:.2f} ± {model_summary['hamiltonian_friction_std']:.2f}")

    # Inertia predictions
    print("\n[4/5] Testing inertia-of-belief predictions...")
    inertia_analyses = analyze_all_subjects(subjects, verbose=False)
    inertia_summary = summarize_inertia_evidence(inertia_analyses)

    print("\n  INERTIA EVIDENCE:")
    print(f"  ─────────────────────────")
    print(f"  Momentum significant: {inertia_summary['momentum_significant_count']}/{inertia_summary['n_subjects']} "
          f"({100*inertia_summary['momentum_significant_rate']:.1f}%)")
    print(f"  Mean update autocorr: {inertia_summary['mean_update_autocorrelation']:.3f}")
    print(f"  Mean overshoot rate: {100*inertia_summary['mean_overshoot_rate']:.1f}%")
    print(f"  LR history-dependent: {inertia_summary['lr_history_dependent_count']}/{inertia_summary['n_subjects']}")
    print(f"  Mean settling time: {inertia_summary['mean_settling_time']:.1f} trials")

    # Generate plots
    print("\n[5/5] Generating plots...")

    # Model comparison
    fig, _ = plot_model_comparison_summary(comparisons)
    fig.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: model_comparison.png")

    # Inertia evidence
    fig, _ = plot_inertia_evidence_summary(inertia_analyses)
    fig.savefig(output_dir / 'inertia_evidence.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: inertia_evidence.png")

    # Example subject trajectory
    example_subject = list(subjects.values())[0]
    example_comparison = comparisons[0]
    fig, _ = plot_belief_trajectory(
        example_subject,
        gradient_fit=example_comparison.gradient_fit,
        hamiltonian_fit=example_comparison.hamiltonian_fit,
        trial_range=(1, 150)
    )
    fig.savefig(output_dir / 'example_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_trajectory.png")

    # Changepoint response
    fig, _ = plot_changepoint_response(example_subject)
    if fig:
        fig.savefig(output_dir / 'changepoint_response.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: changepoint_response.png")

    # Save results to file
    results = {
        'data_summary': data_summary,
        'model_summary': model_summary,
        'inertia_summary': inertia_summary,
    }

    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("INERTIA OF BELIEF - ANALYSIS RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subjects analyzed: {len(subjects)}\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-"*40 + "\n")
        for key, val in model_summary.items():
            f.write(f"  {key}: {val}\n")

        f.write("\nINERTIA EVIDENCE\n")
        f.write("-"*40 + "\n")
        for key, val in inertia_summary.items():
            f.write(f"  {key}: {val}\n")

    print(f"  Saved: results.txt")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Final verdict
    print("\nKEY FINDINGS:")
    if model_summary['hamiltonian_win_rate'] > 0.5:
        print("  ✓ Hamiltonian model fits better than gradient descent")
    else:
        print("  ✗ Gradient descent fits better than Hamiltonian")

    if inertia_summary['momentum_significant_rate'] > 0.5:
        print("  ✓ Significant momentum signature in human updates")
    else:
        print("  ? Weak momentum signature")

    if inertia_summary['mean_overshoot_rate'] > 0.3:
        print("  ✓ Evidence of overshooting after changepoints")
    else:
        print("  ? Limited overshooting observed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run Inertia of Belief analysis on helicopter task data'
    )
    parser.add_argument('--subjects', nargs='+', type=int,
                        help='Subject IDs to analyze (default: all)')
    parser.add_argument('--output', type=str, default='_results/helicopter_analysis',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Skip optimization for faster results')

    args = parser.parse_args()

    run_full_analysis(
        subject_ids=args.subjects,
        output_dir=Path(args.output),
        quick=args.quick
    )


if __name__ == '__main__':
    main()

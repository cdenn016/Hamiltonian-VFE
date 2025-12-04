# -*- coding: utf-8 -*-
"""
Helicopter Task Framework
=========================

Framework for fitting Hamiltonian-VFE models to Nassar's helicopter task data.

Key components:
    data_loader: Load and preprocess helicopter task datasets
    belief_agent: 1D belief agent for task modeling
    corrected_fitting: Corrected model fitting (delta rule, momentum, Hamiltonian)
    analysis: Prediction testing and visualization
"""

from .data_loader import (
    load_mcguire_nassar_2014,
    load_all_datasets,
    HelicopterTrial,
    SubjectData,
    get_subject_ids,
    get_subject_data
)

from .belief_agent import (
    BeliefAgent1D,
    GradientDynamics,
    HamiltonianDynamics
)

from .corrected_fitting import (
    run_delta_rule,
    run_momentum_rule,
    run_hamiltonian,
    fit_all_models,
    FitResult
)

from .analysis import (
    analyze_inertia_predictions,
    analyze_all_subjects,
    summarize_inertia_evidence,
    InertiaAnalysis
)

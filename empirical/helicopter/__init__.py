# -*- coding: utf-8 -*-
"""
Helicopter Task Framework
=========================

Framework for fitting Hamiltonian-VFE models to Nassar's helicopter task data.

Key components:
    data_loader: Load and preprocess helicopter task datasets
    belief_agent: 1D belief agent for task modeling
    dynamics: Hamiltonian vs gradient descent dynamics
    fitting: Parameter fitting and model comparison
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

from .fitting import (
    fit_subject,
    compare_dynamics,
    ModelFit
)

from .analysis import (
    analyze_inertia_predictions,
    analyze_all_subjects,
    summarize_inertia_evidence,
    InertiaAnalysis
)

from .run_analysis import run_full_analysis

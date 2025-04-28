"""
Active Poisoning via Oversampling Unprivileged Group with Adversarial Favorable Samples.

Parameters:
-----------
- dataset: input dataset (aif360-compatible)
- unprivileged_groups: group definitions (e.g., based on protected attributes)
- bp: base rate of the privileged group
- f_label: label indicating a favorable outcome
- model_type: string indicating model type for TModel
- SCALER: preprocessing object (e.g., StandardScaler)
"""

from oversample import group_indices 
from active_select import entropy_select
from models import TModel

import numpy as np

def active_poison(dataset, unprivileged_groups, bp, f_label, model_type, SCALER):
    """
    Selectively flips labels of unprivileged group samples with high uncertainty to simulate poisoning.
    """

    dataset_transf_train = dataset.copy(deepcopy=True)

    # Get indices for unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset, unprivileged_groups)

    unprivileged_dataset = dataset_transf_train.subset(indices)
    privileged_dataset = dataset_transf_train.subset(priv_indices)

    # Count favorable and unfavorable labels in unprivileged group
    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels == f_label)
    n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels != f_label)

    # Number of new favorable samples to simulate
    n_extra_sample = int(bp * len(indices) - n_unpriv_favor)
    print('n_extra:', n_extra_sample)

    # Subset of unprivileged samples with unfavorable label
    x0 = unprivileged_dataset.subset(np.where(unprivileged_dataset.labels != f_label)[0].tolist())
    x0_indices = np.where(unprivileged_dataset.labels != f_label)[0].tolist()
    unpriv_unfav_indices = [indices[i] for i in x0_indices]

    # Train a model and compute prediction probabilities
    test_model = TModel(model_type)
    model = test_model.set_model(dataset, SCALER)
    probs = model.predict_proba(x0.features)

    # Select most uncertain samples
    uncertain_indices = entropy_select(probs, n_extra_sample)
    indices_label_flip = [unpriv_unfav_indices[i] for i in uncertain_indices]

    # Flip labels to favorable outcome
    for i in indices_label_flip:
        dataset_transf_train.labels[i] = [f_label]

    return dataset_transf_train
import sys
import numpy as np
from aif360.metrics import utils
from data_utils import balance
import random


# return dataset indices of unprivileged and privileaged groups
def group_indices (dataset, unprivileged_groups):
    feature_names = dataset.feature_names
    cond_vec = utils.compute_boolean_conditioning_vector(dataset.features, feature_names, unprivileged_groups)

    # indices of examples in the unprivileged and privileged groups
    indices = [i for i, x in enumerate(cond_vec) if x == True]
    priv_indices = [i for i, x in enumerate(cond_vec) if x == False]
    return indices, priv_indices

# oversample unprivileged favored so that the absolute number 
# of the unprivileged favored matches with the privilged favored  
def synthetic_balance (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset.subset(indices) # unprivileaged
    privileged_dataset = dataset.subset(priv_indices) # privilegaed

    # number of unprivileged/privileged with favorable label
    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label
    n_priv_favor = np.count_nonzero(privileged_dataset.labels==f_label) # privileged with favorable label
    n_priv_unfavor = np.count_nonzero(privileged_dataset.labels!=f_label) # privileged with unfavorable label

    # privileged group has more favored in absolute number than the unprivileged group
    if n_unpriv_favor < n_priv_favor:
        # inflate unprivileged favored class
        n_extra_sample = (n_priv_favor - n_unpriv_favor)*sampling_strategy
        if n_extra_sample + n_unpriv_favor >= n_unpriv_unfavor:
            inflate_rate = int(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)
        else:
            inflate_rate = round(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)
        _, extra_favored  = balance(unprivileged_dataset, n_extra_sample, inflate_rate, f_label, uf_label)

        # inflate unprivileged unfavored class to match the privileged baserate
        n_extra_sample = (n_extra_sample + n_unpriv_favor - bp * (n_extra_sample + n_unpriv_favor + n_unpriv_unfavor))/bp
        if n_extra_sample + n_unpriv_unfavor >= n_unpriv_favor:
            inflate_rate = int(((n_extra_sample+n_unpriv_unfavor)/n_unpriv_favor)+1)
        else:
            inflate_rate = round(((n_extra_sample+n_unpriv_unfavor)/n_unpriv_favor)+1)
        _, extra_unfavored  = balance(unprivileged_dataset, n_extra_sample, inflate_rate, uf_label, f_label)

        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, extra_favored.features, extra_unfavored.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, extra_favored.labels, extra_unfavored.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, extra_favored.instance_weights, extra_unfavored.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, extra_favored.protected_attributes, extra_unfavored.protected_attributes))

    return dataset_transf_train


# oversample unfavorable in the privileged group
def synthetic_unfavor_priv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00):

    # [Method 1] inflate privileged unfavored class
    #n_extra_sample = bp_diff * len(priv_indices) * sampling_strategy
    #n_extra_sample = bp_diff * len(priv_indices) / bnp

    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset.subset(indices) # unprivileaged
    privileged_dataset = dataset.subset(priv_indices) # privilegaed

    n_priv_favor = np.count_nonzero(privileged_dataset.labels==f_label) # privileged with favorable label
    n_priv_unfavor = np.count_nonzero(privileged_dataset.labels!=f_label) # privileged with unfavorable label

    n_extra_sample = (n_priv_favor - bnp * len(priv_indices)) / bnp * sampling_strategy

    # privileged unfavorable > privileged favorable
    if n_extra_sample + n_priv_unfavor >= n_priv_favor:
        inflate_rate = int(((n_extra_sample+n_priv_unfavor)/n_priv_favor)+1)
    else:
        inflate_rate = round(((n_extra_sample+n_priv_unfavor)/n_priv_favor)+1)

    dataset_transf_refprivileged_train, extra_unfavored_priv  = balance(privileged_dataset, n_extra_sample, inflate_rate, uf_label, f_label)

    return dataset_transf_refprivileged_train, extra_unfavored_priv


# oversample favorable in the unprivileged group
def synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00):

    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset.subset(indices) # unprivileaged
    privileged_dataset = dataset.subset(priv_indices) # privilegaed

    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label

    n_extra_sample = (bp * len(indices)-n_unpriv_favor) / (1- bp) * sampling_strategy
    # unprivileged favorable > unprivileged unfavorable
    if n_extra_sample + n_unpriv_favor >= n_unpriv_unfavor:
        inflate_rate = int(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)
    else:
        inflate_rate = round(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)

    dataset_transf_refprivileged_train, extra_favored_unpriv  = balance(unprivileged_dataset, n_extra_sample, inflate_rate, f_label, uf_label)

    return dataset_transf_refprivileged_train, extra_favored_unpriv


# adaptive oversampling for the unprivileged group
def synthetic(dataset, unprivileged_groups, bp, bnp, f_label, uf_label, os_mode=2, sampling_strategy=0.50):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # Case 1: privileged group is not relatively favored but has more favored in absolute number than the unprivileged group
    # if privileged base rate is less than unprivilegeted base rate
    if bp < bnp:
        dataset_transf_train = synthetic_balance(dataset, unprivileged_groups, bp, bnp, f_label, uf_label)
        return dataset_transf_train

    # Case 2: if privileged is favored, i.e. has a higher base rate

    # [Method 1] inflate privileged unfavored class
    if os_mode == 1:
        _, sample_unfavor_priv = synthetic_unfavor_priv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_unfavor_priv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_unfavor_priv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_unfavor_priv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_unfavor_priv.protected_attributes))
    elif os_mode == 2:
    # [Method 2] inflate unprivileged favored class
        _, sample_favor_unpriv = synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.0)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_favor_unpriv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_favor_unpriv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_favor_unpriv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_favor_unpriv.protected_attributes))
    # [Method 3] combine methods 1 and 2 
    elif os_mode == 3:
        _, sample_unfavor_priv = synthetic_unfavor_priv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_unfavor_priv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_unfavor_priv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_unfavor_priv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_unfavor_priv.protected_attributes))
        _, sample_favor_unpriv = synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_favor_unpriv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_favor_unpriv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_favor_unpriv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_favor_unpriv.protected_attributes))
    else:
        sys.exit("Oversampling mode is missing: 1: oversample unfavorable privileged; 2: oversample favorable unprivileged; 3. both")

    return dataset_transf_train


def label_bias(dataset, unprivileged_groups, threshold):
    f_label  = dataset.favorable_label
    uf_label = dataset.unfavorable_label
    feature_names = dataset.feature_names

    # make a duplicate copy of the input data
    dataset_label_bias = dataset.copy(deepcopy=True)

    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_label_bias, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset
    #unprivileged_dataset = dataset_label_bias.subset(indices) # unprivileaged
    #privileged_dataset = dataset_label_bias.subset(priv_indices) # privilegaed

    # number of unprivileged/privileged with favorable label
    #n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    #n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label

    #print('Count of unpriv fav before bias injection: ', n_unpriv_favor)
    #print('Count of unpriv unfav before bias injection: ', n_unpriv_unfavor)

    for i in indices:
        if dataset_label_bias.labels[i] == f_label:
            if (random.random() < threshold):
                dataset_label_bias.labels[i] = uf_label

    #unprivileged_dataset = dataset_label_bias.subset(indices) # unprivileaged
    #n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    #n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label
    #print('Count of unpriv fav after bias injection: ', n_unpriv_favor)
    #print('Count of unpriv unfav before bias injection: ', n_unpriv_unfavor)

    return dataset_label_bias


def selection_bias(dataset, unprivileged_groups, threshold):
    f_label  = dataset.favorable_label
    uf_label = dataset.unfavorable_label
    feature_names = dataset.feature_names

    # make a duplicate copy of the input data
    dataset_select_bias = dataset.copy(deepcopy=True)

    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_select_bias, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset
    #unprivileged_dataset = dataset_select_bias.subset(indices) # unprivileaged
    #privileged_dataset = dataset_select_bias.subset(priv_indices) # privilegaed

    # number of unprivileged/privileged with favorable label
    #n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    #n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label
    #n_priv_favor = np.count_nonzero(privileged_dataset.labels==f_label) # privileged with favorable label
    #n_priv_unfavor = np.count_nonzero(privileged_dataset.labels!=f_label) # privileged with unfavorable label
    #print('Count of unpriv fav before bias injection: ', n_unpriv_favor)
    #print('Count of unpriv unfav before bias injection: ', n_unpriv_unfavor)
    #print('Count of priv fav before bias injection: ', n_priv_favor)
    #print('Count of priv unfav before bias injection: ', n_priv_unfavor)
    selected = []
    for i in indices:
        if dataset_select_bias.labels[i] == f_label:
            if (random.random() >= threshold):
                selected.append(i)
        else:
            selected.append(i)

    selected.extend(priv_indices)
    dataset_select_bias = dataset_select_bias.subset(selected)

    #indices, priv_indices = group_indices(dataset_select_bias, unprivileged_groups)
    #unprivileged_dataset = dataset_select_bias.subset(indices) # unprivileaged
    #privileged_dataset = dataset_select_bias.subset(priv_indices) # privilegaed
    #n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    #n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label
    #n_priv_favor = np.count_nonzero(privileged_dataset.labels==f_label) # privileged with favorable label
    #n_priv_unfavor = np.count_nonzero(privileged_dataset.labels!=f_label) # privileged with unfavorable label
    #print('Count of unpriv fav after bias injection: ', n_unpriv_favor)
    #print('Count of unpriv unfav before bias injection: ', n_unpriv_unfavor)
    #print('Count of priv fav before bias injection: ', n_priv_favor)

    return dataset_select_bias


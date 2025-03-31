# oversampling by generating adversarial favored samples in the unprivileged group
# --------------------------------------------------------
# parameters:
# --------------------------------------------------------
# dataset: input data
# unprivileged_groups: definition of unprivileged groups
# bp: base rate of the privileged group
# bnp: base rate of the unprivileged group
# f_label: label of favorate outcome
# --------------------------------------------------------

from oversample import group_indices 
import numpy as np
import random

def adv_synthetic(dataset, unprivileged_groups, bp, bnp, f_label):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # indices of the unprivileged and privileged groups
    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged_dataset and privileged_dataset 
    unprivileged_dataset = dataset_transf_train.subset(indices) # unprivileaged
    privileged_dataset = dataset_transf_train.subset(priv_indices) # privilegaed

    # number of unprivileged with favorable label
    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label

    # number of samples to generate 
    n_extra_sample = (int)((bp * len(indices)-n_unpriv_favor) / (1- bp))
    print('n_extra: ', n_extra_sample)

    # candidate to attack: subset in the unprivileged group with unfavored label
    x0 = unprivileged_dataset.subset(np.where(unprivileged_dataset.labels!=f_label)[0].tolist())
    print('x0: ', x0.features.shape[0])
    
    # random samples from x0 for attacking
    rand_indices = random.sample(range(0, x0.features.shape[0]), n_extra_sample)
    x_os = x0.subset(rand_indices).features
    y_os = x0.subset(rand_indices).labels

    # converting dataset to CDataset format
    from secml.data import CDataset
    ds = CDataset(dataset_transf_train.features, dataset_transf_train.labels.astype(int))
    ds_os = CDataset(x_os, y_os.astype(int))

    # Normalize the data
    from secml.ml.features import CNormalizerMinMax
    nmz = CNormalizerMinMax()
    ds.X = nmz.fit_transform(ds.X)
    ds_os.X = nmz.transform(ds_os.X)

    from secml.ml.classifiers import CClassifierLogistic
    clf = CClassifierLogistic()
    clf.fit(ds.X, ds.Y)

    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    dmax = 0.4  # Maximum perturbation
    lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # parameters of the optimization problem
    solver_params = {
        'eta': 0.3,
        'eta_min': 0.1,
        'eta_max': None,
        'max_iter': 100,
        'eps': 1e-4
    }

    # PGD attack
    from secml.adv.attacks.evasion import CAttackEvasionPGDLS
    pgd_ls_attack = CAttackEvasionPGDLS(
        classifier=clf,
        double_init_ds=ds,
        double_init=False,
        distance=noise_type,
        dmax=dmax,
        lb=lb, ub=ub,
        solver_params=solver_params,
        y_target=y_target)

    # Run the evasion attack on x0 and de_normalize adv_x
    y_pred, _, adv_x, _ = pgd_ls_attack.run(ds_os.X, ds_os.Y)
    adv_x = nmz.inverse_transform(adv_x.X)


    # set weights and protected_attributes for the newly generated samples
    inc = int(n_extra_sample)
    new_weights = [random.choice(dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(dataset.protected_attributes) for _ in range(inc)]

    dataset_transf_train.features = np.concatenate((dataset_transf_train.features,adv_x.tondarray())) 
    dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels,y_pred.tondarray().reshape(-1,1))) 
    dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights,new_weights))  
    dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, new_attributes)) 

    return dataset_transf_train


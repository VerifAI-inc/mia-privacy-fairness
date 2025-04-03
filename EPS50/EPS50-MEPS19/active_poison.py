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
from active_select import entropy_select, uncertainty_select
import numpy as np
import random
import pandas as pd
import numpy as np

# setup test model
from models import TModel

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB

n_est = 1000
min_leaf = 5

def active_poison(dataset, unprivileged_groups, bp, f_label, model_type, SCALER):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # indices of the unprivileged and privileged groups
    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged_dataset and privileged_dataset 
    unprivileged_dataset = dataset_transf_train.subset(indices) # unprivileaged
    privileged_dataset = dataset_transf_train.subset(priv_indices) # privilegaed

    # number of unprivileged with favorable label
    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with favorable label

    # number of samples to generate 
    #n_extra_sample = (int)((bp * len(indices)-n_unpriv_favor) / (1- bp))
    #n_extra_sample = (int)(0.5 * len(indices))
    #n_extra_sample = min((int)(bp * len(indices)-n_unpriv_favor), (int)(n_unpriv_unfavor/2))
    n_extra_sample = (int)(bp * len(indices)-n_unpriv_favor) 
    print('n_extra: ', n_extra_sample)

    # candidate to attack: subset in the unprivileged group with unfavored label
    x0 = unprivileged_dataset.subset(np.where(unprivileged_dataset.labels!=f_label)[0].tolist())
    x0_indices = np.where(unprivileged_dataset.labels!=f_label)[0].tolist()
    unpriv_unfav_indices = [indices[i] for i in x0_indices]
    
    test_model = TModel(model_type)
    model = test_model.set_model(dataset, SCALER)
    #if model_type == 'lr':
    #    model = LogisticRegression(solver='liblinear', random_state=1)
    #elif model_type == 'rf':
    #    model = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf)
    #elif model_type == 'svm':
    #    model = SVC(gamma='auto',probability=True)
    #elif model_type == 'nn':
    #    model = MLPClassifierWrapper(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #elif model_type == 'nb':
    #    model = GaussianNB()
    #model.fit(dataset.features, dataset.labels.ravel())
    probs = model.predict_proba(x0.features)
    #print(pd.DataFrame(model.predict_proba(x0.features), columns=model.classes_))
    #uncertain_indices = uncertainty_select(probs, n_extra_sample) 
    uncertain_indices = entropy_select(probs, n_extra_sample) 
    #print(probs[uncertain_indices])
    indices_label_flip = [unpriv_unfav_indices[i] for i in uncertain_indices] 
    #print(dataset_transf_train.subset(indices_label_flip).labels.ravel())
    new_labels = np.array([f_label]*dataset_transf_train.subset(indices_label_flip).labels.shape[0]) 
    for i in indices_label_flip: 
        dataset_transf_train.labels[i] =[f_label]
    #dataset_transf_train.subset(indices_label_flip).labels = new_labels 
    #print(dataset_transf_train.subset(indices_label_flip).labels.ravel())

    return dataset_transf_train


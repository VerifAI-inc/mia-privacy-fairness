# Metrics function
import numpy as np
from collections import OrderedDict, defaultdict
from aif360.metrics import ClassificationMetric

#setup classification/test models
from models import TModel

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from membership_infer_attack import run_mia_attack

from plot_utils import plot
import matplotlib.pyplot as plt

from oversample import group_indices

import pandas as pd

from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing

# mia2
from privacy_meter.model import Fairlearn_Model, Sklearn_Model, PPModel
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import accuracy_score
from membership_infer_attack import run_mia2_attack
from aif360.datasets import BinaryLabelDataset

from privacy_meter.information_source import InformationSource

def log(y, pre):
    e = 0.0000001
    pre = np.clip(pre, e, 1 - e)
    return - y * np.log(pre) - (1 - y) * np.log(1 - pre)

def test(f_label, uf_label, unprivileged_groups, privileged_groups, dataset, model, thresh_arr, metric_arrs, ATTACK):
    print("=======================================================================")
    print("TEST")
    try:
        #torch classifier
        if "ModelToAttack" in str(type(model)):
            model.eval()
            x = torch.Tensor(dataset.features)
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            with torch.no_grad():
                x = x.to(device)
                pred = model(x)
                y_val_pred_prob = torch.softmax(pred, dim=1)
            pos_ind = int(dataset.favorable_label)
            neg_ind = int(dataset.unfavorable_label)
            
        # sklearn classifier
        else:
            y_val_pred_prob = model.predict_proba(dataset.features)
            
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0] # just returns the favorable value
            neg_ind = np.where(model.classes_ == dataset.unfavorable_label)[0][0] # just returns the unfavorable value
    except AttributeError:
        if (ATTACK == "mia2"):
            # Handle ExponentiatedGradient or in-processing algorithms
            y_val_pred_prob = model._pmf_predict(pd.DataFrame(dataset.features, columns=dataset.feature_names))
        else:
            y_val_pred_prob = model.predict_proba(pd.DataFrame(dataset.features, columns=dataset.feature_names))
            
        unique_labels = [dataset.unfavorable_label, dataset.favorable_label]
        pos_ind = unique_labels.index(dataset.favorable_label)
        neg_ind = unique_labels.index(dataset.unfavorable_label)
    
    print("POS IND", pos_ind)
    print("NEG IND", neg_ind)

    if metric_arrs is None:
        metric_arrs = defaultdict(list)
        
    # print("Lenght of thresh_arr: ", thresh_arr)

    for thresh in thresh_arr:
        # y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        y_val_pred = np.array([0]*y_val_pred_prob.shape[0])
        y_val_pred[np.where(y_val_pred_prob[:,pos_ind] > thresh)[0]] = f_label
        y_val_pred[np.where(y_val_pred_prob[:,pos_ind] <= thresh)[0]] = uf_label
        y_val_pred = y_val_pred.reshape(-1,1)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        # for debug
        # print(f"Accuracy for threshold: {thresh}  is: {metric.accuracy()}")
        # print("Balanced accuracy is: ", (metric.true_positive_rate() + metric.true_negative_rate()) / 2)

        # print other statistics for debugging
        # if len(thresh_arr) == 1:
        #     print("True positive rate is: ", metric.true_positive_rate())
        #     print("True negative rate is: ", metric.true_negative_rate())
        #     print("Balanced accuracy is: ", (metric.true_positive_rate() + metric.true_negative_rate()) / 2)
        #     print("Test Accuracy is: ", metric.accuracy())
        #     print("Positive rate (Unprivileged):", metric.base_rate(privileged=False))
        #     print("Positive rate (Privileged):", metric.base_rate(privileged=True))

        # print("Number of unprivileged instances:", metric.num_instances(privileged=False))
        # print("Number of privileged instances:", metric.num_instances(privileged=True))
        
        # Identify the index of the sensitive feature
        protected_attribute_name = list(privileged_groups[0].keys())[0]
        sensitive_feature_index = dataset.feature_names.index(protected_attribute_name)

        # Unprivileged group positive predictions
        unpriv_indices = np.where(dataset.features[:, sensitive_feature_index] == 0)  # Assuming 0 for unprivileged
        unpriv_positive_predictions = np.sum(y_val_pred[unpriv_indices] == f_label)

        # Privileged group positive predictions
        priv_indices = np.where(dataset.features[:, sensitive_feature_index] == 1)  # Assuming 1 for privileged
        priv_positive_predictions = np.sum(y_val_pred[priv_indices] == f_label)
        
        # Unprivileged group negative predictions
        unpriv_negative_predictions = np.sum(y_val_pred[unpriv_indices] == uf_label)

        # Privileged group negative predictions
        priv_negative_predictions = np.sum(y_val_pred[priv_indices] == uf_label)

        # Print the results
        print(f"Unprivileged Positive Predictions (Favorable): {unpriv_positive_predictions}")
        print(f"Privileged Positive Predictions (Favorable): {priv_positive_predictions}")
        
        print(f"Unprivileged Negative Predictions: {unpriv_negative_predictions}")
        print(f"Privileged Negative Predictions: {priv_negative_predictions}") 
        
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(1 - min((metric.disparate_impact()), 1/metric.disparate_impact()))
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
        metric_arrs['unpriv_fpr'].append(metric.false_positive_rate(privileged=False))
        metric_arrs['unpriv_fnr'].append(metric.false_negative_rate(privileged=False))
        metric_arrs['priv_fpr'].append(metric.false_positive_rate(privileged=True))
        metric_arrs['priv_fnr'].append(metric.false_negative_rate(privileged=True))
        
    print("=======================================================================")

    return metric_arrs


def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    metrics,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    if metrics is None:
        metrics = defaultdict(list)

        metrics['bal_acc'] = 0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate())
        metrics['avg_odds_diff'] = classified_metric_pred.average_odds_difference()
        metrics['disp_imp'] = 1-min(classified_metric_pred.disparate_impact(), 1/classified_metric_pred.disparate_impact())
        metrics['stat_par_diff'] = classified_metric_pred.statistical_parity_difference()
        metrics['eq_opp_diff'] = classified_metric_pred.equal_opportunity_difference()
        metrics['theil_ind'] = classified_metric_pred.theil_index()
        metrics['unpriv_fpr'].append(classified_metric_pred.false_positive_rate(privileged=False))
        metrics['unpriv_fnr'].append(classified_metric_pred.false_negative_rate(privileged=False))
        metrics['priv_fpr'].append(classified_metric_pred.false_positive_rate(privileged=True))
        metrics['priv_fnr'].append(classified_metric_pred.false_negative_rate(privileged=True))
    else:
        metrics['bal_acc'].append(0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate()))
        metrics['avg_odds_diff'].append(classified_metric_pred.average_odds_difference()) 
        metrics['disp_imp'].append(1-min(classified_metric_pred.disparate_impact(), 1/classified_metric_pred.disparate_impact()))
        metrics['stat_par_diff'].append(classified_metric_pred.statistical_parity_difference())
        metrics['eq_opp_diff'].append(classified_metric_pred.equal_opportunity_difference())
        metrics['theil_ind'].append(classified_metric_pred.theil_index())
        metrics['unpriv_fpr'].append(classified_metric_pred.false_positive_rate(privileged=False))
        metrics['unpriv_fnr'].append(classified_metric_pred.false_negative_rate(privileged=False))
        metrics['priv_fpr'].append(classified_metric_pred.false_positive_rate(privileged=True))
        metrics['priv_fnr'].append(classified_metric_pred.false_negative_rate(privileged=True))
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics

def describe_metrics(metrics, thresh_arr, TEST=True):
    if not TEST:
        best_ind = np.argmax(metrics['bal_acc'])
        # print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    else:
        best_ind = -1
    # print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    # print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    # print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    # print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    # print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    # print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
    # print("Corresponding false positive_rate for privileged: {:6.4f}".format(metrics['priv_fpr'][best_ind]))
    # print("Corresponding false negative_rate for privileged: {:6.4f}".format(metrics['priv_fnr'][best_ind]))
    # print("Corresponding false positive_rate for unpribileged: {:6.4f}".format(metrics['unpriv_fpr'][best_ind]))
    # print("Corresponding false negative_rate for unprivileged: {:6.4f}".format(metrics['unpriv_fnr'][best_ind]))
    
def calculate_accuracy(model, dataset):
    if "ModelToAttack" in str(type(model)):
        model.eval()
        x = torch.Tensor(dataset.features)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            y_val_pred_prob = torch.softmax(pred, dim=1)
            y_pred = y_val_pred_prob.max(1).indices
        y_true = dataset.labels.ravel()
        print("Classification report for train: ")
        print(classification_report(y_true, y_pred))

        return sum(y_pred.numpy() == y_true)/len(y_pred)

    else:
        X = pd.DataFrame(
                dataset.features, columns=dataset.feature_names
            )         
        
        y_pred = model.predict(X)
        y_true = dataset.labels.ravel()
        print("Classification report for train: ")
        print(classification_report(y_true, y_pred))
        
        return sum(y_pred == y_true)/len(y_pred)

def get_info_sources(target_dataset, reference_dataset, model):
    target_info_source = InformationSource(
        models=[model],
        datasets=[target_dataset]
    )
    
    reference_info_source = InformationSource(
        models=[model],
        datasets=[reference_dataset]
    )
    
    return target_info_source, reference_info_source

def get_test_metrics_for_eg(target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train
    X = pd.DataFrame(
        dataset.features, columns=dataset.feature_names
    )    
    y_true = dataset.labels.ravel() 
    sens_attr = dataset.protected_attribute_names[0]  
    sensitive_features = dataset.features[:, dataset.feature_names.index(sens_attr)]
    constraint = EqualizedOdds(difference_bound=0.001)
    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=10)
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        mitigator = ExponentiatedGradientReduction(sens_attr, classifier, constraint)
        mitigator.fit(X, y_true)        
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mitigator)
    elif ATTACK == "mia2":
        mitigator = ExponentiatedGradient(classifier, constraint)
        mitigator.fit(X, y_true, sensitive_features=sensitive_features)        
        
        target_model = Fairlearn_Model(model_obj=mitigator, loss_fn=log)
        
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
        
    print("####Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mitigator, dataset))
        
    # find the best threshold for balanced accuracy
    print('Validating EG ...')
    
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        
        
    if (f_label != None and uf_label != None):
        val_metrics = test(f_label, uf_label,
                        unprivileged_groups, privileged_groups,
                        dataset=dataset_orig_val_pred,
                        model=mitigator,
                        thresh_arr=thresh_arr, metric_arrs=None, ATTACK=ATTACK)
        
        orig_best_ind = np.argmax(val_metrics['bal_acc'])
        
        # for debugging
        print("Best thresh: ", thresh_arr[orig_best_ind])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                val_metrics['avg_odds_diff'], 'avg. odds diff.')

            plt.show()

        describe_metrics(val_metrics, thresh_arr)

        print('Testing EG ...')
        test_metrics = test(f_label, uf_label,
                            unprivileged_groups, privileged_groups,
                            dataset=dataset_orig_test_pred,
                            model=mitigator,
                            # select thereshold based on best balanced accuracy
                            thresh_arr=[thresh_arr[orig_best_ind]], 
                            # 0.5
                            # thresh_arr=[thresh_arr[-1]], 
                            metric_arrs=test_metrics, ATTACK=ATTACK)

        describe_metrics(test_metrics, thresh_arr)
        
    for i in results:
        print(i)
        
    # metrics array to hold the results
    if mia_metrics is None:
        mia_metrics = defaultdict(list)

    # Add the results to test_metrics object
    # MIA results for overall dataset and subpopulations
    for i in range(len(results)):
        mia_metrics[f"{results[i].get_name()}_mia_auc"].append(results[i].get_auc())
        mia_metrics[f"{results[i].get_name()}_mia_privacy_risk"].append(results[i].get_privacy_risk())
        mia_metrics[f"{results[i].get_name()}_mia_ppv"].append(results[i].get_ppv())
        mia_metrics[f"{results[i].get_name()}_mia_attacker_advantage"].append(results[i].get_attacker_advantage())
        mia_metrics[f"{results[i].get_name()}_mia_result"].append(results[i])

    return test_metrics, mia_metrics

def get_test_metrics_for_syn_rew(target_dataset, reference_dataset, syn_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, mia_metrics, ATTACK, log_type, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
    """
    Return the metrics for sysntetic mitigator, the difference being the fact that it additionally recieves orig train for running MIA
    """
    dataset = syn_dataset

    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER, ATTACK)
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mod_orig)
    elif ATTACK == "mia2":
        target_model = Sklearn_Model(model_obj=mod_orig, loss_fn=log)
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
    
    print("Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
    
    # find the best threshold for balanced accuracy
    print('Validating Syn OR Rew ...')
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    val_metrics = test(f_label, uf_label,
                       unprivileged_groups, privileged_groups,
                       dataset=dataset_orig_val_pred,
                       model=mod_orig,
                       thresh_arr=thresh_arr, metric_arrs=None, ATTACK=ATTACK)
    
    orig_best_ind = np.argmax(val_metrics['bal_acc'])
    # for debugging
    print("Best thresh: ", thresh_arr[orig_best_ind])
 

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

    if DISPLAY:
        plot(thresh_arr, model_type + ' Original Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             disp_imp_err, '1 - min(DI, 1/DI)')

        plot(thresh_arr, model_type + ' Original Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             val_metrics['avg_odds_diff'], 'avg. odds diff.')

        plt.show()

    describe_metrics(val_metrics, thresh_arr)

    print('Testing Syn OR Rew ...')
    test_metrics = test(f_label, uf_label,
                        unprivileged_groups, privileged_groups,
                        dataset=dataset_orig_test_pred,
                        model=mod_orig,
                        # select thereshold based on best balanced accuracy
                        thresh_arr=[thresh_arr[orig_best_ind]], 
                        # 0.5
                        # thresh_arr=[thresh_arr[-1]], 
                        metric_arrs=test_metrics, ATTACK=ATTACK)

    describe_metrics(test_metrics, thresh_arr)
            
    for i in results:
        print(i)
        
    # metrics array to hold the results
    if mia_metrics is None:
        mia_metrics = defaultdict(list)

    # Add the results to test_metrics object
    # MIA results for overall dataset and subpopulations
    for i in range(len(results)):
        mia_metrics[f"{results[i].get_name()}_mia_auc"].append(results[i].get_auc())
        mia_metrics[f"{results[i].get_name()}_mia_privacy_risk"].append(results[i].get_privacy_risk())
        mia_metrics[f"{results[i].get_name()}_mia_ppv"].append(results[i].get_ppv())
        mia_metrics[f"{results[i].get_name()}_mia_attacker_advantage"].append(results[i].get_attacker_advantage())
        mia_metrics[f"{results[i].get_name()}_mia_result"].append(results[i])

    return test_metrics, mia_metrics


def print_cpp_accuracies(dataset_orig_train, dataset_orig_test, 
                         train_pred_cpp, test_pred_cpp, 
                         unprivileged_groups, f_label, uf_label):
    """
    Print accuracies for different subgroups before and after post-processing
    using scores to derive predicted labels.
    """

    def calculate_subset_accuracy(dataset, predictions):
        # Derive predicted labels from predictions.scores using a threshold
        threshold = 0.5
        # If scores have two columns: second column is probability of favorable label.
        # If one column, assume it is the probability of favorable label directly.
        if predictions.scores.ndim == 2 and predictions.scores.shape[1] == 2:
            pred_prob = predictions.scores[:, 1]
        else:
            pred_prob = predictions.scores.ravel()

        pred_labels = np.where(pred_prob >= threshold, 
                               predictions.favorable_label, 
                               predictions.unfavorable_label).reshape(-1, 1)

        # Ensure shapes match
        assert dataset.labels.shape[0] == pred_labels.shape[0], \
            f"Mismatch in shapes: dataset={dataset.labels.shape}, pred={pred_labels.shape}"

        return np.mean(dataset.labels.ravel() == pred_labels.ravel())

    # Function to get group indices
    def group_indices(dataset, unprivileged_groups):
        # This function should return two lists/arrays of indices:
        # indices for unprivileged groups and indices for privileged groups.
        # Implement it as per your original code/environment.
        # For example:
        cond_unpriv = np.ones(len(dataset.labels), dtype=bool)
        for grp in unprivileged_groups:
            for attr, val in grp.items():
                col_idx = dataset.protected_attribute_names.index(attr)
                cond_unpriv &= (dataset.protected_attributes[:, col_idx] == val)
        unpriv_indices = np.where(cond_unpriv)[0]
        priv_indices = np.setdiff1d(np.arange(len(dataset.labels)), unpriv_indices)
        return unpriv_indices, priv_indices

    train_indices, train_priv_indices = group_indices(dataset_orig_train, unprivileged_groups)
    test_indices, test_priv_indices = group_indices(dataset_orig_test, unprivileged_groups)
    
    def create_subsets(dataset, indices, priv_indices, f_label, uf_label):
        unprivileged_dataset = dataset.subset(indices)
        privileged_dataset = dataset.subset(priv_indices)

        uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
        f_unpriv_indices  = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
        uf_priv_indices   = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
        f_priv_indices    = np.where(privileged_dataset.labels.ravel() == f_label)[0]

        # Return subsets and their indices so we can apply them identically to predictions
        return (
            unprivileged_dataset.subset(uf_unpriv_indices),
            unprivileged_dataset.subset(f_unpriv_indices),
            privileged_dataset.subset(uf_priv_indices),
            privileged_dataset.subset(f_priv_indices),
            uf_unpriv_indices, f_unpriv_indices, uf_priv_indices, f_priv_indices
        )
    
    # Create train subsets and indices
    (train_uf_unpriv, train_f_unpriv, train_uf_priv, train_f_priv, 
     train_uf_unpriv_idx, train_f_unpriv_idx, train_uf_priv_idx, train_f_priv_idx) = create_subsets(
        dataset_orig_train, train_indices, train_priv_indices, f_label, uf_label
    )

    # Create test subsets and indices
    (test_uf_unpriv, test_f_unpriv, test_uf_priv, test_f_priv,
     test_uf_unpriv_idx, test_f_unpriv_idx, test_uf_priv_idx, test_f_priv_idx) = create_subsets(
        dataset_orig_test, test_indices, test_priv_indices, f_label, uf_label
    )
    
    # Subset predicted datasets using the exact same indices
    train_unpriv_pred = train_pred_cpp.subset(train_indices)
    train_priv_pred = train_pred_cpp.subset(train_priv_indices)

    test_unpriv_pred = test_pred_cpp.subset(test_indices)
    test_priv_pred = test_pred_cpp.subset(test_priv_indices)

    # Use previously obtained indices directly
    train_pred_uf_unpriv = train_unpriv_pred.subset(train_uf_unpriv_idx)
    train_pred_f_unpriv = train_unpriv_pred.subset(train_f_unpriv_idx)
    train_pred_uf_priv = train_priv_pred.subset(train_uf_priv_idx)
    train_pred_f_priv = train_priv_pred.subset(train_f_priv_idx)

    test_pred_uf_unpriv = test_unpriv_pred.subset(test_uf_unpriv_idx)
    test_pred_f_unpriv = test_unpriv_pred.subset(test_f_unpriv_idx)
    test_pred_uf_priv = test_priv_pred.subset(test_uf_priv_idx)
    test_pred_f_priv = test_priv_pred.subset(test_f_priv_idx)

    # Calculate accuracies using derived predictions
    train_results = {
        "uf_unpriv_cpp": calculate_subset_accuracy(train_uf_unpriv, train_pred_uf_unpriv),
        "f_unpriv_cpp": calculate_subset_accuracy(train_f_unpriv, train_pred_f_unpriv),
        "uf_priv_cpp": calculate_subset_accuracy(train_uf_priv, train_pred_uf_priv),
        "f_priv_cpp": calculate_subset_accuracy(train_f_priv, train_pred_f_priv)
    }

    test_results = {
        "uf_unpriv_cpp": calculate_subset_accuracy(test_uf_unpriv, test_pred_uf_unpriv),
        "f_unpriv_cpp": calculate_subset_accuracy(test_f_unpriv, test_pred_f_unpriv),
        "uf_priv_cpp": calculate_subset_accuracy(test_uf_priv, test_pred_uf_priv),
        "f_priv_cpp": calculate_subset_accuracy(test_f_priv, test_pred_f_priv)
    }
    
    # Print results in a formatted table
    print("Accuracies Comparison:")
    print("-" * 70)
    print(f"{'Group':<15}{'CPP Train':<20}{'CPP Test':<20}")
    print("-" * 70)
    print(f"{'UF Unpriv':<15}{train_results['uf_unpriv_cpp']:<20.4f}{test_results['uf_unpriv_cpp']:<20.4f}")
    print(f"{'UF Priv':<15}{train_results['uf_priv_cpp']:<20.4f}{test_results['uf_priv_cpp']:<20.4f}")
    print(f"{'F Unpriv':<15}{train_results['f_unpriv_cpp']:<20.4f}{test_results['f_unpriv_cpp']:<20.4f}")
    print(f"{'F Priv':<15}{train_results['f_priv_cpp']:<20.4f}{test_results['f_priv_cpp']:<20.4f}")
    print("-" * 70)
    
    return train_results, test_results

def get_test_metrics_for_cpp(target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train
    dataset_orig_test, dataset_orig_val = dataset_orig_test.split([0.5], shuffle=True)
    
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    
    if SCALER:
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset.features)
        X_test = scale_orig.fit_transform(dataset_orig_test.features)
        X_valid = scale_orig.fit_transform(dataset_orig_val.features)
    else:
        X_train = dataset.features
        X_test = dataset_orig_test.features
        X_valid = dataset_orig_val.features
        
    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER, ATTACK)
    
    fav_idx = np.where(mod_orig.classes_ == dataset.favorable_label)[0][0]
    y_train_pred_prob = mod_orig.predict_proba(X_train)[:,fav_idx]
    y_valid_pred_prob = mod_orig.predict_proba(X_valid)[:,fav_idx]
    y_test_pred_prob = mod_orig.predict_proba(X_test)[:,fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = np.column_stack([
        1 - y_train_pred_prob,  # Probability for class 0
        y_train_pred_prob       # Probability for class 1
    ])

    dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
    # Assign two columns: probability for class 0 and class 1
    dataset_orig_test_pred.scores = np.column_stack([
        1 - y_test_pred_prob,  # Probability for class 0
        y_test_pred_prob       # Probability for class 1
    ])

    y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    dataset_orig_valid_pred.labels = y_valid_pred
        
    # Assign predicted labels based on the threshold
    y_train_pred = np.where(y_train_pred_prob >= class_thresh, 
                            dataset_orig_train_pred.favorable_label, 
                            dataset_orig_train_pred.unfavorable_label)
    dataset_orig_train_pred.labels = y_train_pred.reshape(-1, 1)

    y_test_pred = np.where(y_test_pred_prob >= class_thresh, 
                           dataset_orig_test_pred.favorable_label, 
                           dataset_orig_test_pred.unfavorable_label)
    dataset_orig_test_pred.labels = y_test_pred.reshape(-1, 1)

    # Initialize the CalibratedEqualizedOdds post-processor
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint="fnr",
                                     seed=12345679)
    cpp = cpp.fit(dataset_orig_val, dataset_orig_valid_pred)
    
    # **Key change**: Predict once on entire datasets, not per subgroup
    train_pred_cpp = cpp.predict(dataset_orig_train_pred)
    test_pred_cpp = cpp.predict(dataset_orig_test_pred)

    # Print accuracies
    train_results, test_results = print_cpp_accuracies(
        dataset_orig_train, dataset_orig_test, 
        train_pred_cpp, test_pred_cpp,
        unprivileged_groups, 
        f_label, uf_label
    )
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type + "_cpp", cpp)
    elif ATTACK == "mia2":
        target_model = PPModel(model_obj=cpp, loss_fn=log, feature_columns=dataset.feature_names, protected_attribute_name='age', label_name=dataset.label_names[0])
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
    
    print("####Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
        
    # find the best threshold for balanced accuracy
    print('Validating Original ...')
    
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        
    if (f_label != None and uf_label != None):
        val_metrics = test(f_label, uf_label,
                        unprivileged_groups, privileged_groups,
                        dataset=dataset_orig_val_pred,
                        model=mod_orig,
                        thresh_arr=thresh_arr, metric_arrs=None, ATTACK=ATTACK)
        
        orig_best_ind = np.argmax(val_metrics['bal_acc'])
        
        # for debugging
        print("Best thresh: ", thresh_arr[orig_best_ind])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                val_metrics['avg_odds_diff'], 'avg. odds diff.')

            plt.show()

        describe_metrics(val_metrics, thresh_arr)

        print('Testing Original ...')
        test_metrics = test(f_label, uf_label,
                            unprivileged_groups, privileged_groups,
                            dataset=dataset_orig_test_pred,
                            model=mod_orig,
                            # select thereshold based on best balanced accuracy
                            thresh_arr=[thresh_arr[orig_best_ind]], 
                            # 0.5
                            # thresh_arr=[thresh_arr[-1]], 
                            metric_arrs=test_metrics, ATTACK=ATTACK)

        describe_metrics(test_metrics, thresh_arr)
        
    for i in results:
        print(i)
        
    # metrics array to hold the results
    if mia_metrics is None:
        mia_metrics = defaultdict(list)

    # Add the results to test_metrics object
    # MIA results for overall dataset and subpopulations
    for i in range(len(results)):
        mia_metrics[f"{results[i].get_name()}_mia_auc"].append(results[i].get_auc())
        mia_metrics[f"{results[i].get_name()}_mia_privacy_risk"].append(results[i].get_privacy_risk())
        mia_metrics[f"{results[i].get_name()}_mia_ppv"].append(results[i].get_ppv())
        mia_metrics[f"{results[i].get_name()}_mia_attacker_advantage"].append(results[i].get_attacker_advantage())
        mia_metrics[f"{results[i].get_name()}_mia_result"].append(results[i])

    return test_metrics, mia_metrics

def get_test_metrics(target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train

    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER, ATTACK)
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mod_orig)
    elif ATTACK == "mia2":
        target_model = Sklearn_Model(model_obj=mod_orig, loss_fn=log)
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
    
    print("####Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
        
    # find the best threshold for balanced accuracy
    print('Validating Original ...')
    
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        
        
    if (f_label != None and uf_label != None):
        val_metrics = test(f_label, uf_label,
                        unprivileged_groups, privileged_groups,
                        dataset=dataset_orig_val_pred,
                        model=mod_orig,
                        thresh_arr=thresh_arr, metric_arrs=None, ATTACK=ATTACK)
        
        orig_best_ind = np.argmax(val_metrics['bal_acc'])
        
        # for debugging
        print("Best thresh: ", thresh_arr[orig_best_ind])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, model_type + ' Original Classification Thresholds',
                val_metrics['bal_acc'], 'Balanced Accuracy',
                val_metrics['avg_odds_diff'], 'avg. odds diff.')

            plt.show()

        describe_metrics(val_metrics, thresh_arr)

        print('Testing Original ...')
        test_metrics = test(f_label, uf_label,
                            unprivileged_groups, privileged_groups,
                            dataset=dataset_orig_test_pred,
                            model=mod_orig,
                            # select thereshold based on best balanced accuracy
                            thresh_arr=[thresh_arr[orig_best_ind]], 
                            # 0.5
                            # thresh_arr=[thresh_arr[-1]], 
                            metric_arrs=test_metrics, ATTACK=ATTACK)

        describe_metrics(test_metrics, thresh_arr)
        
    for i in results:
        print(i)
        
    # metrics array to hold the results
    if mia_metrics is None:
        mia_metrics = defaultdict(list)

    # Add the results to test_metrics object
    # MIA results for overall dataset and subpopulations
    for i in range(len(results)):
        mia_metrics[f"{results[i].get_name()}_mia_auc"].append(results[i].get_auc())
        mia_metrics[f"{results[i].get_name()}_mia_privacy_risk"].append(results[i].get_privacy_risk())
        mia_metrics[f"{results[i].get_name()}_mia_ppv"].append(results[i].get_ppv())
        mia_metrics[f"{results[i].get_name()}_mia_attacker_advantage"].append(results[i].get_attacker_advantage())
        mia_metrics[f"{results[i].get_name()}_mia_result"].append(results[i])

    return test_metrics, mia_metrics

def get_orig_model_metrics(dataset_orig_train, dataset_orig_test, unprivileged_groups, f_label, uf_label, model_type, SCALER, ATTACK):
    results = {}
    
    # TRAINING ACCURACIES
    # Fit the model
    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset_orig_train, SCALER, ATTACK)
    
    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_orig_train, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset_orig_train.subset(indices) # unprivileaged
    privileged_dataset = dataset_orig_train.subset(priv_indices) # privilegaed
    
    # subgroups
    uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
    uf_unpriv_dataset = unprivileged_dataset.subset(uf_unpriv_indices)
    f_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
    f_unpriv_dataset = unprivileged_dataset.subset(f_unpriv_indices)
    uf_priv_indices = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
    uf_priv_dataset = privileged_dataset.subset(uf_priv_indices)
    f_priv_indices = np.where(privileged_dataset.labels.ravel() == f_label)[0]
    f_priv_dataset = privileged_dataset.subset(f_priv_indices)
    
    results["train_0_0"] = calculate_accuracy(mod_orig, uf_unpriv_dataset)
    results["train_1_0"] = calculate_accuracy(mod_orig, uf_priv_dataset)
    results["train_0_1"] = calculate_accuracy(mod_orig, f_unpriv_dataset)
    results["train_1_1"] = calculate_accuracy(mod_orig, f_priv_dataset)
    
    # TESTING ACCURACIES
    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_orig_test, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset_orig_test.subset(indices) # unprivileaged
    privileged_dataset = dataset_orig_test.subset(priv_indices) # privilegaed
    
    # subgroups
    uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
    uf_unpriv_dataset = unprivileged_dataset.subset(uf_unpriv_indices)
    f_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
    f_unpriv_dataset = unprivileged_dataset.subset(f_unpriv_indices)
    uf_priv_indices = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
    uf_priv_dataset = privileged_dataset.subset(uf_priv_indices)
    f_priv_indices = np.where(privileged_dataset.labels.ravel() == f_label)[0]
    f_priv_dataset = privileged_dataset.subset(f_priv_indices)
    
    results["test_0_0"] = calculate_accuracy(mod_orig, uf_unpriv_dataset)
    results["test_1_0"] = calculate_accuracy(mod_orig, uf_priv_dataset)
    results["test_0_1"] = calculate_accuracy(mod_orig, f_unpriv_dataset)
    results["test_1_1"] = calculate_accuracy(mod_orig, f_priv_dataset)
    
    # Print accuracies as a table
    print("Accuracies:")
    print("-" * 50)
    print(f"{'Group':<15}{'Train Accuracy':<20}{'Test Accuracy':<20}")
    print("-" * 50)
    print(f"{'0_-':<15}{results['train_0_0']:<20.4f}{results['test_0_0']:<20.4f}")
    print(f"{'1_-':<15}{results['train_1_0']:<20.4f}{results['test_1_0']:<20.4f}")
    print(f"{'0_+':<15}{results['train_0_1']:<20.4f}{results['test_0_1']:<20.4f}")
    print(f"{'1_+':<15}{results['train_1_1']:<20.4f}{results['test_1_1']:<20.4f}")
    print("-" * 50)
    
    return results

def get_egr_model_metrics(dataset_orig_train, dataset_orig_test, unprivileged_groups, f_label, uf_label, egr_mod, SCALER):
    results = {}
    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_orig_train, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset_orig_train.subset(indices) # unprivileaged
    privileged_dataset = dataset_orig_train.subset(priv_indices) # privilegaed
    
    # subgroups
    uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
    uf_unpriv_dataset = unprivileged_dataset.subset(uf_unpriv_indices)
    f_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
    f_unpriv_dataset = unprivileged_dataset.subset(f_unpriv_indices)
    uf_priv_indices = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
    uf_priv_dataset = privileged_dataset.subset(uf_priv_indices)
    f_priv_indices = np.where(privileged_dataset.labels.ravel() == f_label)[0]
    f_priv_dataset = privileged_dataset.subset(f_priv_indices)
    
    results["train_0_0"] = calculate_accuracy(egr_mod, uf_unpriv_dataset)
    results["train_0_1"] = calculate_accuracy(egr_mod, f_unpriv_dataset)
    results["train_1_0"] = calculate_accuracy(egr_mod, uf_priv_dataset)
    results["train_1_1"] = calculate_accuracy(egr_mod, f_priv_dataset)
    
    # TESTING ACCURACIES
    # indices of examples in the unprivileged and privileged groups
    indices, priv_indices = group_indices(dataset_orig_test, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset_orig_test.subset(indices) # unprivileaged
    privileged_dataset = dataset_orig_test.subset(priv_indices) # privilegaed
    
    # subgroups
    uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
    uf_unpriv_dataset = unprivileged_dataset.subset(uf_unpriv_indices)
    f_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
    f_unpriv_dataset = unprivileged_dataset.subset(f_unpriv_indices)
    uf_priv_indices = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
    uf_priv_dataset = privileged_dataset.subset(uf_priv_indices)
    f_priv_indices = np.where(privileged_dataset.labels.ravel() == f_label)[0]
    f_priv_dataset = privileged_dataset.subset(f_priv_indices)
    
    results["test_0_0"] = calculate_accuracy(egr_mod, uf_unpriv_dataset)
    results["test_0_1"] = calculate_accuracy(egr_mod, f_unpriv_dataset)
    results["test_1_0"] = calculate_accuracy(egr_mod, uf_priv_dataset)
    results["test_1_1"] = calculate_accuracy(egr_mod, f_priv_dataset)
    
    return results
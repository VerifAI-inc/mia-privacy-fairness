# Metrics function
import numpy as np
from collections import OrderedDict, defaultdict
from aif360.metrics import ClassificationMetric

import shap

from models import MLPClassifierWithWeightWrapper

#setup classification/test models
from models import TModel

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

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
import diffprivlib.models as dp

from privacy_meter.information_source import InformationSource

from sklearn.metrics import confusion_matrix

def log(y, pre):
    e = 0.0000001
    pre = np.clip(pre, e, 1 - e)
    return - y * np.log(pre) - (1 - y) * np.log(1 - pre)

def test(f_label, uf_label, unprivileged_groups, privileged_groups, dataset, model, thresh_arr, metric_arrs):
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
            print("Checking predict proba")
            y_val_pred_prob = model.predict_proba(dataset.features)
            
            # pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0] # just returns the favorable value
            # neg_ind = np.where(model.classes_ == dataset.unfavorable_label)[0][0] # just returns the unfavorable value
            unique_labels = [dataset.unfavorable_label, dataset.favorable_label]
            pos_ind = unique_labels.index(dataset.favorable_label)
            neg_ind = unique_labels.index(dataset.unfavorable_label)
    except AttributeError:
        # if (ATTACK == "mia2"):
            # Handle ExponentiatedGradient or in-processing algorithms
        y_val_pred_prob = model._pmf_predict(pd.DataFrame(dataset.features, columns=dataset.feature_names))
        # else:
        #     y_val_pred_prob = model.predict_proba(pd.DataFrame(dataset.features, columns=dataset.feature_names))
            
        unique_labels = [dataset.unfavorable_label, dataset.favorable_label]
        pos_ind = unique_labels.index(dataset.favorable_label)
        neg_ind = unique_labels.index(dataset.unfavorable_label)
    
    # print("POS IND", pos_ind)
    # print("NEG IND", neg_ind)

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
        
        disp_impact = metric.disparate_impact()

        if np.isnan(disp_impact) or disp_impact == 0:
            disp_impact = np.mean(metric_arrs['disp_imp']) if metric_arrs['disp_imp'] else 1  # Default to 1

        # Prevent ZeroDivisionError
        if disp_impact == 0:
            metric_arrs['disp_imp'].append(1)  # Default value if DI is zero
        else:
            metric_arrs['disp_imp'].append(1 - min(disp_impact, 1/disp_impact))

        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
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
    if metrics['disp_imp'][best_ind] == 0:
        disp_imp_at_best_ind = 1  # Assign a safe default value
    else:
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

def get_test_metrics_for_eg(DATASET, target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train
    X = dataset.features
    y_true = dataset.labels.ravel() 
    sens_attr = dataset.protected_attribute_names[0]  
    sensitive_features = dataset.features[:, dataset.feature_names.index(sens_attr)]
    
    lower_bounds = 0
    upper_bounds = 1
    
    constraint = EqualizedOdds(difference_bound=0.001)
    if model_type == 'dt':
        classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=10)
    elif model_type == 'dplr':
        classifier = dp.LogisticRegression(solver='liblinear', random_state=1, epsilon=10, bounds=(lower_bounds, upper_bounds))
    elif model_type == 'lr':
        classifier = LogisticRegression(solver='liblinear', random_state=1)
    elif model_type == 'dprf':
        if DATASET == "bank" or DATASET.startswith("german") or DATASET == "meps19":
            classifier = dp.RandomForestClassifier(random_state=1,  epsilon=10, bounds=(lower_bounds, upper_bounds), max_depth=15)
        elif DATASET.startswith("compas"):
            classifier = dp.RandomForestClassifier(random_state=1,  epsilon=10, bounds=(lower_bounds, upper_bounds), max_depth=6)
        elif DATASET == "law_sex" or DATASET == "law_race":
            classifier = dp.RandomForestClassifier(random_state=1,  epsilon=10, bounds=(lower_bounds, upper_bounds), max_depth=7)
        elif DATASET.startswith("law"):
            classifier = dp.RandomForestClassifier(random_state=1,  epsilon=10, bounds=(lower_bounds, upper_bounds), max_depth=3)
    elif model_type == 'rf':
        if DATASET == "bank" or DATASET.startswith("german") or DATASET == "meps19":
            classifier = RandomForestClassifier(random_state=1, max_depth=15)
        elif DATASET.startswith("compas"):
            classifier = RandomForestClassifier(random_state=1, max_depth=6)
        elif DATASET == "law_sex" or DATASET == "law_race":
            classifier = RandomForestClassifier(random_state=1, max_depth=7)
        elif DATASET.startswith("law"):
            classifier = RandomForestClassifier(random_state=1, max_depth=3)
    elif model_type == 'mlp':
        classifier = MLPClassifierWithWeightWrapper()
    mitigator = ExponentiatedGradient(classifier, constraint)
    mitigator.fit(X, y_true, sensitive_features=sensitive_features)
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type + "_egr", mitigator)
    elif ATTACK == "mia2":
        target_model = Fairlearn_Model(model_obj=mitigator, loss_fn=log)
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
    
    # print("####Train metrics:")
    # print("Train accuracy: ", calculate_accuracy(dataset))
        
    # find the best threshold for balanced accuracy
    # print('Validating EG ...')
    
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
                        thresh_arr=thresh_arr, metric_arrs=None)
        
        orig_best_ind = np.argmax(val_metrics['bal_acc'])
        
        # for debugging
        # print("Best thresh: ", thresh_arr[orig_best_ind])

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
                            metric_arrs=test_metrics)

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
        
    test_metrics = append_accuracies_to_metrics(test_metrics, dataset, dataset_orig_test, mitigator, f_label, uf_label, unprivileged_groups)

    # # **NEW: Compute SHAP values and store in test_metrics**
    # try:
    #     explainer = shap.Explainer(mitigator.predict, dataset.features)  # Use a black-box SHAP explainer
    #     shap_values = explainer(dataset_orig_test.features)

    #     # Compute mean absolute SHAP values for each feature
    #     mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    #     # Store SHAP values in test_metrics
    #     test_metrics["shap_values"].append(mean_shap_values.tolist())

    # except Exception as e:
    #     print(f"⚠️ SHAP computation failed: {e}")
    #     test_metrics["shap_values"].append([np.nan] * len(dataset.feature_names))

    return test_metrics, mia_metrics

def get_test_metrics_for_syn_rew(DATASET, target_dataset, reference_dataset, syn_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, mia_metrics, ATTACK, log_type, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
    """
    Return the metrics for sysntetic mitigator, the difference being the fact that it additionally recieves orig train for running MIA
    """
    dataset = syn_dataset

    test_model = TModel(DATASET, model_type)
    if (model_type == 'mlp' and log_type == 'rew_log'):
        mod_orig = test_model.set_model(dataset, dataset.instance_weights, ATTACK)
    elif (model_type == 'mlp'):
        mod_orig = test_model.set_model(dataset, None, ATTACK)
    else:
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
    
    # print("Train metrics:")
    # print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
    
    # # find the best threshold for balanced accuracy
    # print('Validating Syn OR Rew ...')
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
                       thresh_arr=thresh_arr, metric_arrs=None)
    
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
                        metric_arrs=test_metrics)

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
        
    test_metrics = append_accuracies_to_metrics(test_metrics, dataset, dataset_orig_test, mod_orig, f_label, uf_label, unprivileged_groups)
    
    #  # **NEW: Compute SHAP values and store in test_metrics**
    # try:
    #     explainer = shap.Explainer(mod_orig.predict, dataset.features)  # Use a black-box SHAP explainer
    #     shap_values = explainer(dataset_orig_test.features)

    #     # Compute mean absolute SHAP values for each feature
    #     mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    #     # Store SHAP values in test_metrics
    #     test_metrics["shap_values"].append(mean_shap_values.tolist())

    # except Exception as e:
    #     print(f"⚠️ SHAP computation failed: {e}")
    #     test_metrics["shap_values"].append([np.nan] * len(dataset.feature_names))

    return test_metrics, mia_metrics

def get_test_metrics_for_cpp(target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train
    
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        
    if SCALER:
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset.features)
        X_test = scale_orig.transform(dataset_orig_test.features)
        X_valid = scale_orig.transform(dataset_orig_val.features)
    else:
        X_train = dataset.features
        X_test = dataset_orig_test.features
        X_valid = dataset_orig_val.features
        
    test_model = TModel(DATASET, model_type)
    if (model_type == 'mlp'):
        mod_orig = test_model.set_model(dataset, None, ATTACK)
    else:
        mod_orig = test_model.set_model(dataset, SCALER, ATTACK)
    
    fav_idx = np.where(mod_orig.classes_ == dataset.favorable_label)[0][0]
    y_train_pred_prob = mod_orig.predict_proba(X_train)[:,fav_idx]
    y_valid_pred_prob = mod_orig.predict_proba(X_valid)[:,fav_idx]
    y_test_pred_prob = mod_orig.predict_proba(X_test)[:,fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
    dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)
    
    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    dataset_orig_valid_pred.labels = y_valid_pred
    
    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred

    # Initialize the CalibratedEqualizedOdds post-processor
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint="fnr",
                                     seed=12345679)
    cpp = cpp.fit(dataset_orig_val, dataset_orig_valid_pred)
    
    # Predict once on entire datasets, not per subgroup
    train_pred_cpp = cpp.predict(dataset_orig_train_pred)
    test_pred_cpp = cpp.predict(dataset_orig_test_pred)
    
    if ATTACK == "mia1":
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type + "_cpp", cpp)
    elif ATTACK == "mia2":
        target_model = PPModel(model_obj=cpp, loss_fn=log, feature_columns=dataset.feature_names, protected_attribute_name=dataset.protected_attribute_names[0], label_name=dataset.label_names[0])
        target_info_source, reference_info_source = get_info_sources(target_dataset, reference_dataset, target_model)
        _, _, _, pop_metrics, results = run_mia2_attack(target_info_source, reference_info_source, log_type)
        thresh_arr = pop_metrics['thresholds']
    
    # print("####Train metrics:")
    # print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
        
    # # find the best threshold for balanced accuracy
    # print('Validating Original ...')
    
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
                        thresh_arr=thresh_arr, metric_arrs=None)
        
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
                            metric_arrs=test_metrics)

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
    
    test_metrics = print_cpp_accuracies(test_metrics, dataset, dataset_orig_test, train_pred_cpp, test_pred_cpp, unprivileged_groups, f_label, uf_label)

    # if hasattr(mod_orig, "feature_importances_"):  # Ensure model supports feature importance
    #     feature_importance = mod_orig.feature_importances_.tolist()
    #     test_metrics["feature_importances"].append(feature_importance)

    return test_metrics, mia_metrics

def get_test_metrics(DATASET, target_dataset, reference_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, 
                     test_metrics, mia_metrics, ATTACK, log_type, f_label=None, uf_label=None, 
                     unprivileged_groups=None, privileged_groups=None, THRESH_ARR=None, DISPLAY=None, SCALER=None):
    dataset = dataset_orig_train

    test_model = TModel(DATASET, model_type)
    if (model_type == 'mlp'):
        mod_orig = test_model.set_model(dataset, None, ATTACK)
    else:
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
    
    # print("####Train metrics:")
    # print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
        
    # # find the best threshold for balanced accuracy
    # print('Validating Original ...')
    
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
                        thresh_arr=thresh_arr, metric_arrs=None)
        
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
                            metric_arrs=test_metrics)

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
        
    test_metrics = append_accuracies_to_metrics(test_metrics, dataset, dataset_orig_test, mod_orig, f_label, uf_label, unprivileged_groups)

    # # **NEW: Compute SHAP values and store in test_metrics**
    # try:
    #     explainer = shap.Explainer(mod_orig.predict, dataset.features)  # Use a black-box SHAP explainer
    #     shap_values = explainer(dataset_orig_test.features)

    #     # Compute mean absolute SHAP values for each feature
    #     mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    #     # Store SHAP values in test_metrics
    #     test_metrics["shap_values"].append(mean_shap_values.tolist())

    # except Exception as e:
    #     print(f"⚠️ SHAP computation failed: {e}")
    #     test_metrics["shap_values"].append([np.nan] * len(dataset.feature_names))

    return test_metrics, mia_metrics

def calculate_accuracy(dataset, model):
    X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    y_pred = model.predict(X)
    y_true = dataset.labels.ravel()
    accuracy = sum(y_pred == y_true) / len(y_pred)
    return accuracy

def append_accuracies_to_metrics(test_metrics, train_dataset, test_dataset, model, f_label, uf_label, unprivileged_groups):
    """
    Calculate and append train/test accuracies (overall and subgroup-specific) to the test_metrics dictionary.

    Args:
        test_metrics (dict): The dictionary to append accuracy metrics.
        train_dataset (BinaryLabelDataset): The training dataset.
        test_dataset (BinaryLabelDataset): The testing dataset.
        model (object): The model used for predictions.
        log_type (str): The log type indicating the mitigator (e.g., 'un_log', 'syn_log', etc.).
        f_label (int): Favorable label.
        uf_label (int): Unfavorable label.
        unprivileged_groups (list): List of unprivileged group definitions.

    Returns:
        dict: Updated test_metrics dictionary with added accuracy metrics.
    """
    def calculate_subset_accuracies(dataset, indices, f_label, uf_label, model):
        unpriv_indices, priv_indices = indices
        
        unpriv_indices = np.array(unpriv_indices)
        priv_indices = np.array(priv_indices)

        # Subgroups within unprivileged and privileged groups
        unpriv_uf_indices = unpriv_indices[np.where(dataset.labels[unpriv_indices].ravel() == uf_label)[0]]
        unpriv_f_indices = unpriv_indices[np.where(dataset.labels[unpriv_indices].ravel() == f_label)[0]]
        priv_uf_indices = priv_indices[np.where(dataset.labels[priv_indices].ravel() == uf_label)[0]]
        priv_f_indices = priv_indices[np.where(dataset.labels[priv_indices].ravel() == f_label)[0]]
        
        unpriv_uf_df = pd.DataFrame(dataset.features[unpriv_uf_indices], columns=dataset.feature_names)
        unpriv_f_df = pd.DataFrame(dataset.features[unpriv_f_indices], columns=dataset.feature_names)
        priv_uf_df = pd.DataFrame(dataset.features[priv_uf_indices], columns=dataset.feature_names)
        priv_f_df = pd.DataFrame(dataset.features[priv_f_indices], columns=dataset.feature_names)
    
         # Predictions
        unpriv_uf_preds = model.predict(unpriv_uf_df)
        unpriv_f_preds = model.predict(unpriv_f_df)
        priv_uf_preds = model.predict(priv_uf_df)
        priv_f_preds = model.predict(priv_f_df)

        # Accuracy calculations
        return {
            "0_-": sum(unpriv_uf_preds == dataset.labels[unpriv_uf_indices].ravel()) / len(unpriv_uf_preds),
            "0_+": sum(unpriv_f_preds == dataset.labels[unpriv_f_indices].ravel()) / len(unpriv_f_preds),
            "1_-": sum(priv_uf_preds == dataset.labels[priv_uf_indices].ravel()) / len(priv_uf_preds),
            "1_+": sum(priv_f_preds == dataset.labels[priv_f_indices].ravel()) / len(priv_f_preds)
        }

    # Calculate overall accuracies
    train_accuracy = calculate_accuracy(train_dataset, model)
    test_accuracy = calculate_accuracy(test_dataset, model)

    # Group indices for subpopulation metrics
    train_indices = group_indices(train_dataset, unprivileged_groups)
    test_indices = group_indices(test_dataset, unprivileged_groups)

    train_subgroup_accuracies = calculate_subset_accuracies(train_dataset, train_indices, f_label, uf_label, model)
    test_subgroup_accuracies = calculate_subset_accuracies(test_dataset, test_indices, f_label, uf_label, model)

    # Append results to test_metrics
    test_metrics[f"accuracy_train"].append(train_accuracy)
    test_metrics[f"accuracy_test"].append(test_accuracy)

    for subgroup, accuracy in train_subgroup_accuracies.items():
        test_metrics[f"accuracy_train_{subgroup}"].append(accuracy)
    for subgroup, accuracy in test_subgroup_accuracies.items():
        test_metrics[f"accuracy_test_{subgroup}"].append(accuracy)

    return test_metrics

def print_cpp_accuracies(test_metrics, dataset_orig_train, dataset_orig_test, 
                         train_pred_cpp, test_pred_cpp, 
                         unprivileged_groups, f_label, uf_label):
    def calculate_overall_accuracy(dataset, predictions):
        # Derive predicted labels
        threshold = 0.5
        pred_labels = np.where(predictions.scores.ravel() >= threshold, 
                               predictions.favorable_label, 
                               predictions.unfavorable_label).reshape(-1, 1)
        # Ensure shapes match
        assert dataset.labels.shape[0] == pred_labels.shape[0], \
            f"Mismatch in shapes: dataset={dataset.labels.shape}, pred={pred_labels.shape}"
        return np.mean(dataset.labels.ravel() == pred_labels.ravel())

    def calculate_subset_metrics(subset_dataset, subset_predictions):
        # Derive predicted labels
        threshold = 0.5
        pred_prob = subset_predictions.scores.ravel()
        pred_labels = np.where(pred_prob >= threshold, 
                               subset_predictions.favorable_label, 
                               subset_predictions.unfavorable_label).reshape(-1, 1)
        # Ensure shapes match
        assert subset_dataset.labels.shape[0] == pred_labels.shape[0], \
            f"Mismatch in shapes: dataset={subset_dataset.labels.shape}, pred={pred_labels.shape}"
        return np.mean(subset_dataset.labels.ravel() == pred_labels.ravel())

    def create_subsets(dataset, indices, priv_indices, f_label, uf_label):
        unprivileged_dataset = dataset.subset(indices)
        privileged_dataset = dataset.subset(priv_indices)

        uf_unpriv_indices = np.where(unprivileged_dataset.labels.ravel() == uf_label)[0]
        f_unpriv_indices  = np.where(unprivileged_dataset.labels.ravel() == f_label)[0]
        uf_priv_indices   = np.where(privileged_dataset.labels.ravel() == uf_label)[0]
        f_priv_indices    = np.where(privileged_dataset.labels.ravel() == f_label)[0]

        return (
            unprivileged_dataset.subset(uf_unpriv_indices),
            unprivileged_dataset.subset(f_unpriv_indices),
            privileged_dataset.subset(uf_priv_indices),
            privileged_dataset.subset(f_priv_indices),
            uf_unpriv_indices, f_unpriv_indices, uf_priv_indices, f_priv_indices
        )

    def create_and_compute_metrics(dataset, predictions, indices, priv_indices):
        # Create subsets
        subsets = create_subsets(dataset, indices, priv_indices, f_label, uf_label)
        subset_results = {}
        for subset_name, subset_data, pred_data in zip(
            ["0_-", "0_+", "1_-", "1_+"],
            subsets[:4],
            [predictions.subset(idx) for idx in subsets[4:]]
        ):
            subset_results[f"{subset_name}"] = calculate_subset_metrics(subset_data, pred_data)
        return subset_results

    # Overall accuracies
    test_metrics["accuracy_train"].append(calculate_overall_accuracy(dataset_orig_train, train_pred_cpp))
    test_metrics["accuracy_test"].append(calculate_overall_accuracy(dataset_orig_test, test_pred_cpp))

    # Subpopulation metrics for train and test datasets
    train_indices, train_priv_indices = group_indices(dataset_orig_train, unprivileged_groups)
    test_indices, test_priv_indices = group_indices(dataset_orig_test, unprivileged_groups)

    train_metrics = create_and_compute_metrics(dataset_orig_train, train_pred_cpp, train_indices, train_priv_indices)
    for k, v in train_metrics.items():
        test_metrics[f"accuracy_train_{k}"].append(v)

    test_metrics_data = create_and_compute_metrics(dataset_orig_test, test_pred_cpp, test_indices, test_priv_indices)
    for k, v in test_metrics_data.items():
        test_metrics[f"accuracy_test_{k}"].append(v)

    return test_metrics

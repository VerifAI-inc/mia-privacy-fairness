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

def test(f_label, uf_label, unprivileged_groups, privileged_groups, dataset, model, thresh_arr, metric_arrs):
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
            
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
            neg_ind = np.where(model.classes_ == dataset.unfavorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
        neg_ind = 1
        print('y_val_pred_prob: ', y_val_pred_prob)

    if metric_arrs is None:
        metric_arrs = defaultdict(list)

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
        
        # print other statistics for debugging
        if len(thresh_arr) == 1:
            print("True positive rate is: ", metric.true_positive_rate())
            print("True negative rate is: ", metric.true_negative_rate())
            print("Balanced accuracy is: ", (metric.true_positive_rate() + metric.true_negative_rate()) / 2)
            print("Test Accuracy is: ", metric.accuracy())
        else:
            # for debug
            print(f"Accuracy for threshold: {thresh}  is: {metric.accuracy()}")
            print("Balanced accuracy is: ", (metric.true_positive_rate() + metric.true_negative_rate()) / 2)
    
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(1 - min((metric.disparate_impact()), 1/metric.disparate_impact()))
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
        metric_arrs['unpriv_fpr'].append(metric.false_positive_rate(privileged=False))
        metric_arrs['unpriv_fnr'].append(metric.false_negative_rate(privileged=False))
        metric_arrs['priv_fpr'].append(metric.false_positive_rate(privileged=True))
        metric_arrs['priv_fnr'].append(metric.false_negative_rate(privileged=True))

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
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    else:
        best_ind = -1
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
    print("Corresponding false positive_rate for privileged: {:6.4f}".format(metrics['priv_fpr'][best_ind]))
    print("Corresponding false negative_rate for privileged: {:6.4f}".format(metrics['priv_fnr'][best_ind]))
    print("Corresponding false positive_rate for unpribileged: {:6.4f}".format(metrics['unpriv_fpr'][best_ind]))
    print("Corresponding false negative_rate for unprivileged: {:6.4f}".format(metrics['unpriv_fnr'][best_ind]))
    
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
        y_pred = model.predict(dataset.features)
        y_true = dataset.labels.ravel()
        print("Classification report for train: ")
        print(classification_report(y_true, y_pred))
        
        return sum(y_pred == y_true)/len(y_pred)

        



def get_test_metrics_for_syn(syn_dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
    """
    Return the metrics for sysntetic mitigator, the difference being the fact that it additionally recieves orig train for running MIA
    """
    dataset = syn_dataset

    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER)
    
    
    print("Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
    
    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

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
                        # thresh_arr=[thresh_arr[orig_best_ind]], 
                        # 0.5
                        thresh_arr=[thresh_arr[-1]], 
                        metric_arrs=test_metrics)

    describe_metrics(test_metrics, thresh_arr)
    
    # Runnning MIA attack based on original train for syntetic 
    results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mod_orig)
        
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



def get_test_metrics(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):

    dataset = dataset_orig_train

    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER)
    
    
    print("####Train metrics:")
    print("Train accuracy: ", calculate_accuracy(mod_orig, dataset))
    
    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

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
    
    # Runnning MIA attack based on subgroups
    results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mod_orig)
        
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
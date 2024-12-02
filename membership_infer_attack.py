#####################################################################
"""Functions and Modules to Perform Membership Inference Attacks"""
#####################################################################
"""
    # per-example losses:
    # loss_train  shape: (n_train, )
    # loss_test  shape: (n_test, )
    loss_train = np.array([0.1, 0.3, 0.4, 0.2, 0.4, 0.2, 0.1,
                           1.2, 1.3, 1.4, 1.5])
    loss_test = np.array([0.9, 1, 0.9, 1.2, 1.4, 0.8, 0.9,
                          1.7, 1.8, 1.9, 1.7])

    # labels denote class types
    labels_train = np.array([0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1])

    labels_test = np.array([0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1])
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from MIA_Attack_Result import MIA_Attack_Result
from scipy.special import xlogy
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import warnings


#setup classification/test models
from models import TModel

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report

# mia2
from privacy_meter.information_source import InformationSource
from privacy_meter import audit_report
from privacy_meter.audit_report import *
from privacy_meter.audit import Audit, MetricEnum

from sklearn.metrics import auc

import matplotlib.pyplot as plt

# Suppose we have evaluated the model on training and test examples to get the
# per-example losses:
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )
"""
loss_train = np.array([0.1, 0.3, 0.4, 0.2, 0.4, 0.2, 0.1,
                       1.2, 1.3, 1.4, 1.5])
loss_test = np.array([0.9, 1, 0.9, 1.2, 1.4, 0.8, 0.9,
                      1.7, 1.8, 1.9, 1.7])

# labels denote class types
labels_train = np.array([0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1])

labels_test = np.array([0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1])
"""

# logistic loss funtion
def compute_loss_lr(y, y_pred):
    return -(xlogy(y, y_pred) + xlogy(1 - y, 1 - y_pred))

def compute_loss_dt(x_prob):
    return 1-x_prob*x_prob - (1 - x_prob)*(1 - x_prob)

def log_losses(y_true, y_pred, eps=1e-15):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    one_minus_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(one_minus_y_pred))

def analyze_and_visualize_predictions_and_losses(preds_train, preds_test, loss_train, loss_test):
    """
    Analyze and visualize predictions and losses for training and testing datasets.
    
    Args:
        preds_train (pd.DataFrame): DataFrame containing training predictions (columns: ["0", "1", "label"]).
        preds_test (pd.DataFrame): DataFrame containing testing predictions (columns: ["0", "1", "label"]).
        loss_train (np.ndarray): Array of loss values for training data.
        loss_test (np.ndarray): Array of loss values for testing data.
    """
    # Statistical summaries
    print("\n=== Statistical Summary of Predictions and Losses ===")
    
    # Predictions summary
    print("\nTrain Predictions (Positive Class Probability):")
    print(preds_train["1"].describe())
    print("\nTest Predictions (Positive Class Probability):")
    print(preds_test["1"].describe())

    # Losses summary
    print("\nTraining Loss Summary:")
    print(pd.DataFrame(loss_train, columns=['Loss']).describe())
    print("\nTesting Loss Summary:")
    print(pd.DataFrame(loss_test, columns=['Loss']).describe())

    # Visualization

    # Convert losses to DataFrame for visualization
    loss_train_df = pd.DataFrame({'loss': loss_train, 'dataset': 'train'})
    loss_test_df = pd.DataFrame({'loss': loss_test, 'dataset': 'test'})
    loss_combined = pd.concat([loss_train_df, loss_test_df], ignore_index=True)

    # Plot histogram of predicted probabilities (train)
    plt.figure(figsize=(12, 5))
    plt.hist(preds_train["1"], bins=30, alpha=0.7, label="Train Predictions (Positive Class)", color='blue')
    plt.hist(preds_test["1"], bins=30, alpha=0.7, label="Test Predictions (Positive Class)", color='orange')
    plt.title("Histogram of Predicted Probabilities for Positive Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot scatter plot of true labels vs. predicted probabilities
    plt.figure(figsize=(12, 5))
    plt.scatter(preds_train["label"], preds_train["1"], alpha=0.6, label="Train Data", color='blue')
    plt.scatter(preds_test["label"], preds_test["1"], alpha=0.6, label="Test Data", color='orange')
    plt.title("True Labels vs. Predicted Probabilities")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.show()

    # Plot histogram of losses
    plt.figure(figsize=(12, 5))
    for dataset, group in loss_combined.groupby("dataset"):
        plt.hist(group["loss"], bins=30, alpha=0.7, label=f"{dataset.capitalize()} Loss")
    plt.title("Histogram of Loss Values")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Compare loss distributions with box plots
    plt.figure(figsize=(8, 6))
    loss_combined.boxplot(column='loss', by='dataset', grid=False)
    plt.title("Box Plot of Loss Values by Dataset")
    plt.suptitle("")  # Remove automatic title
    plt.xlabel("Dataset")
    plt.ylabel("Loss")
    plt.show()

def run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, mod_orig ):
    print("=================================================================")
    print("RUN MIA ATTACK")
    
    # getting the column denoting priv/unpriv groups
    priv_group_column = list(privileged_groups[0].keys())[0]

    # create df for train
    df_train = pd.DataFrame(dataset_orig_train.features, columns=dataset_orig_train.feature_names)
    df_train["labels"] = dataset_orig_train.labels
    
    # create labels train where labels denote class types, in compas it is race
    # labels_train = df_train[priv_group_column]
    protected_attr_train = df_train[priv_group_column]

    # subgroup for + and -
    subgroup_labels_train = df_train["labels"]

    # the same steps for test
    df_test = pd.DataFrame(dataset_orig_test.features, columns=dataset_orig_test.feature_names)
    df_test["labels"] = dataset_orig_test.labels

    # labels_test = df_test[priv_group_column]
    protected_attr_test = df_test[priv_group_column]
    # subgroup for + and -
    subgroup_labels_test = df_test["labels"]

    # Getting per example loss for train/test dataset
    if model_type == "dt":
        if hasattr(mod_orig, "predict_proba"):  # Handles DT case
            preds_train = pd.DataFrame(mod_orig.predict_proba(dataset_orig_train.features), columns=["0", "1"])
        elif hasattr(mod_orig, "_pmf_predict"):  # Handles EG case
            preds_train = pd.DataFrame(mod_orig._pmf_predict(pd.DataFrame(dataset_orig_train.features)), columns=["0", "1"])
        
        # Add true labels to the predictions
        preds_train["label"] = dataset_orig_train.labels

        # Calculate per-example loss for training data
        loss_train = log_losses(preds_train["label"], preds_train["1"])
        
        if hasattr(mod_orig, "predict_proba"):  # Handles DT case
            preds_test = pd.DataFrame(mod_orig.predict_proba(dataset_orig_test.features), columns=["0", "1"])
        elif hasattr(mod_orig, "_pmf_predict"):  # Handles EG case
            preds_test = pd.DataFrame(mod_orig._pmf_predict(pd.DataFrame(dataset_orig_test.features)), columns=["0", "1"])
        
        # Add true labels to the predictions
        preds_test["label"] = dataset_orig_test.labels

        # Calculate per-example loss for testing data
        loss_test = log_losses(preds_test["label"], preds_test["1"])   
        
        analyze_and_visualize_predictions_and_losses(preds_train, preds_test, loss_train, loss_test)
    else: #model_type == "lr" or model_type == "nn":
        # per example loss for train
        if model_type != "nn":
            preds_train = pd.DataFrame(mod_orig.predict_proba(dataset_orig_train.features), columns=["0", "1"])
            preds_train["label"] = dataset_orig_train.labels
            #loss_train = compute_loss_lr(preds_train["label"], preds_train["1"])
            loss_train = log_losses(preds_train["label"], preds_train["1"])

            # per example loss for test
            preds_test = pd.DataFrame(mod_orig.predict_proba(dataset_orig_test.features), columns=["0", "1"])
            preds_test["label"] = dataset_orig_test.labels
            # loss_test = compute_loss_lr(preds_test["label"], preds_test["1"])   
            loss_test = log_losses(preds_test["label"], preds_test["1"])
        # torch NN
        else:
            
            # calculating losses for train
            mod_orig.eval()
            x = torch.Tensor(dataset_orig_train.features)
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            with torch.no_grad():
                x = x.to(device)
                pred = mod_orig(x)
                y_val_pred_prob = torch.softmax(pred, dim=1)
                y_val_pred_prob = y_val_pred_prob.numpy()
                
            preds_train = pd.DataFrame(y_val_pred_prob, columns=["0", "1"])
            
            preds_train["label"] = dataset_orig_train.labels
            # loss_train = compute_loss_lr(preds_train["label"], preds_train["1"])
            loss_train = log_losses(preds_train["label"], preds_train["1"])

            # calculating loss for test
            x = torch.Tensor(dataset_orig_test.features)
            
            with torch.no_grad():
                x = x.to(device)
                pred = mod_orig(x)
                y_val_pred_prob = torch.softmax(pred, dim=1)
                y_val_pred_prob = y_val_pred_prob.numpy()
            
            preds_test = pd.DataFrame(y_val_pred_prob, columns=["0", "1"])
            preds_test["label"] = dataset_orig_test.labels
            # loss_test = compute_loss_lr(preds_test["label"], preds_test["1"]) 
            loss_test = log_losses(preds_test["label"], preds_test["1"]) 
     
    #else:
    #    raise NotImplementedError(f"Loss function for {model_type} should be implemented!")
        
    # making the length of loss train and loss test equal
    # consider this for syntetic mitigator because it increases dataset size artificially
    """
    if len(loss_test) < len(loss_train):
        loss_train = loss_train[:len(loss_test)]
        labels_train = labels_train[:len(protected_attr_test)]
    """
    #######################
        
    results = run_mia_attacks_against_vulnarablle_subpopulations(loss_train, loss_test, protected_attr_train, protected_attr_test,
                                                                 subgroup_labels_train, subgroup_labels_test)
    return results
    
    
def run_threshold_mia_attack(name, loss_train,loss_test):
    # Run the attack based on the thresholds 
    ntrain = len(loss_train) # 5
    ntest = len(loss_test) # 5
    fpr, tpr, thresholds = roc_curve(
        np.concatenate((np.ones(ntrain), np.zeros(ntest))), # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # roc_curve uses classifier in the form of
        # "score >= threshold ==> predict positive", while training data has lower
        # loss, so we negate the loss.
        -np.concatenate((loss_train, loss_test))
        )
    auc_score = roc_auc_score( np.concatenate((np.ones(ntrain), np.zeros(ntest))),  -np.concatenate((loss_train, loss_test)))
    test_train_ratio = ntest / ntrain
    
    # Calculate attack accuracy
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]
    
    tpr_ = sum(loss_train <= optimal_threshold)/ntrain
    tnr = sum(loss_test > optimal_threshold)/ntest
    overall_acc = (sum(loss_train <= optimal_threshold) + sum(loss_test > optimal_threshold)) / (ntrain + ntest)
    privacy_risk = 0.5*(tpr_+tnr)
    
    return MIA_Attack_Result(name, fpr, tpr, thresholds, auc_score, privacy_risk, 
                             overall_acc, tpr_, tnr, test_train_ratio, [ntrain, ntest])


def run_threshold_estimator(
    name, 
    losses_train,
    losses_test,
    method="best_loss_threshold",
    eval_sets=None,
    microdata=False,
    enforce_uniform_prior=True,
):
    """
    Estimate privacy of a given model using a threshold attack.
    """
    
    if method == "best_loss_threshold":
        threshold = 0
        best_acc = 0
        for c in itertools.chain(losses_train, losses_test):
            if enforce_uniform_prior:
                # Compute balanced accuracy.
                acc = 0.5 * ((losses_train < c).mean() + (losses_test >= c).mean())
            else:
                acc = np.concatenate([losses_train < c, losses_test >= c]).mean()

            if acc > best_acc:
                threshold = c
                best_acc = acc

    elif method == "average_loss_threshold":
        threshold = losses_train.mean()

    else:
        raise NotImplementedError(method)

    # Eval sets.
    if eval_sets is not None:
        eval_y_train, eval_preds_train, eval_y_test, eval_preds_test = eval_sets
        eval_losses_train = log_losses(eval_y_train, eval_preds_train)
        eval_losses_test = log_losses(eval_y_test, eval_preds_test)
    else:
        eval_losses_train = losses_train
        eval_losses_test = losses_test

    in_guesses = eval_losses_train < threshold
    out_guesses = eval_losses_test >= threshold
    if microdata:
        if enforce_uniform_prior:
            warnings.warn(_MSG_ENFORCE_UNIFORM_PRIOR_AND_MICRODATA)
        index=list(y_train.index) + list(y_test.index)
        return pd.Series(np.concatenate([in_guesses, out_guesses]), index=index)

    else:
        if enforce_uniform_prior:
            overall_acc =  0.5 * (in_guesses.mean() + out_guesses.mean())
        else:
            overall_acc = np.concatenate([in_guesses, out_guesses]).mean()
            
    # Run the attack based on the thresholds 
    ntrain = len(losses_train)
    ntest = len(losses_test)
    
    print(f"Number of training samples (ntrain): {ntrain}")
    print(f"Number of test samples (ntest): {ntest}")

    
    fpr, tpr, thresholds = roc_curve(
        np.concatenate((np.ones(ntrain), np.zeros(ntest))),
        # roc_curve uses classifier in the form of
        # "score >= threshold ==> predict positive", while training data has lower
        # loss, so we negate the loss.
        -np.concatenate((losses_train, losses_test))
        )
    auc_score = roc_auc_score( np.concatenate((np.ones(ntrain), np.zeros(ntest))),  -np.concatenate((losses_train, losses_test)))
    test_train_ratio = ntest / ntrain
    """
    # Calculate attack accuracy
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]
    
    tpr_ = sum(loss_train <= optimal_threshold)/ntrain
    tnr = sum(loss_test > optimal_threshold)/ntest
    overall_acc = (sum(loss_train <= optimal_threshold) + sum(loss_test > optimal_threshold)) / (ntrain + ntest)
    privacy_risk = 0.5*(tpr_+tnr)
    
    """
    # we have already calculated optimal threshold
    optimal_threshold = threshold
    
    tpr_ = sum(losses_train < optimal_threshold)/ntrain
    tnr = sum(losses_test >= optimal_threshold)/ntest
    # overall_acc = (sum(losses_train < optimal_threshold) + sum(losses_test >= optimal_threshold)) / (ntrain + ntest)
    overall_acc =  0.5 * (in_guesses.mean() + out_guesses.mean())
    privacy_risk = 0.5*(tpr_+tnr)
    
    return MIA_Attack_Result(name, fpr, tpr, thresholds, auc_score, privacy_risk, 
                             overall_acc, tpr_, tnr, test_train_ratio, [ntrain, ntest])

        
def select_data_by_class(train, test, protected_attr_train,  protected_attr_test, protected_attr_val,
                         labels_train, labels_test, unique_label):
    """
    Select particular subgroup protected attribute combination from train and test dataset
    """
    return train[(protected_attr_train == protected_attr_val) & (labels_train == unique_label)], test[(protected_attr_test == protected_attr_val) & (labels_test == unique_label)]


def run_mia_attacks_against_vulnarablle_subpopulations(loss_train, loss_test, protected_attr_train, protected_attr_test, labels_train, labels_test):
    results = []
    mia_res = run_threshold_mia_attack("entire_dataset",  loss_train, loss_test)
    results.append(mia_res)
    
    # running the  attack against each subpopulation
    protected_attr_vals = np.unique(protected_attr_train)
    unique_labels = np.unique(labels_train)
    
    for unique_label in unique_labels:
        mia_res = run_threshold_mia_attack("entire_dataset_label_" + str(unique_label),
                                           loss_train[labels_train == unique_label],
                                           loss_test[labels_test == unique_label])
        results.append(mia_res)

    for protected_attr_val in protected_attr_vals:
        for unique_label in unique_labels:
            print("Protected_attr_val, label,", protected_attr_val, unique_label)
            train_subpopulation, test_subpopulation = select_data_by_class(loss_train,  loss_test,                                           
                                                                           protected_attr_train, protected_attr_test, protected_attr_val,
                                                                           labels_train, labels_test, unique_label)
            # 1) sub_pop_mia_res = run_threshold_mia_attack("subpopulation_" + str(subgroup) + "_label_"+str(unique_sub_class),
            #                                          train_subpopulation, test_subpopulation)
            
            sub_pop_mia_res = run_threshold_estimator("subpopulation_" + str(protected_attr_val) + "_label_"+str(unique_label),
                                           train_subpopulation, test_subpopulation)
            results.append(sub_pop_mia_res)
    return results

## EPFL implementation

def run_mia2_attack(target_info_source, reference_info_source, log_type):
    print("========================================================================")
    print("RUN MIA2 ATTACK")
    audit_obj = Audit(
        metrics=MetricEnum.GROUPPOPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        logs_directory_names=log_type + "_group"
    )
    
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    metric_results, group_metrics = audit_results[0], audit_results[1]
    
    pop_audit_obj = Audit(
        metrics=MetricEnum.POPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        logs_directory_names=log_type + "_pop"
    )
    
    pop_audit_obj.prepare()
    pop_audit_results = pop_audit_obj.run()[0]
    pop_metric_results, pop_metrics = pop_audit_results[0], pop_audit_results[1]
    
    results = []
    
    dataset_size = [len(pop_metrics['member_signals']), len(pop_metrics['non_member_signals'])]
    mia_res = MIA_Attack_Result(
        name="entire_dataset",
        fpr=pop_metrics['fprs'],
        tpr=pop_metrics['tprs'],
        thresholds=pop_metrics['thresholds'],
        auc_score=auc(pop_metrics['fprs'], pop_metrics['tprs']),
        privacy_risk=(pop_metrics['tpr'] + pop_metrics['tnr']) / 2,
        accuracy=pop_metrics['accuracy'],
        tpr_ind=pop_metrics['tpr'],
        tnr_ind=pop_metrics['tnr'],
        test_train_ratio=len(pop_metrics['non_member_signals']) / len(pop_metrics['member_signals']),
        dataset_size=dataset_size,
    )
    
    results.append(mia_res)
    
    for _, group in enumerate(group_metrics.keys()):
        if group != "overall_thresh_arr":
            metrics = group_metrics[group]
            
            # Extract privacy risk and AUC for unfair model
            privacy_risk = (metrics['tpr'] + metrics['tnr']) / 2
            auc_score = auc(metrics['fprs'], metrics['tprs'])
            
            if group == 2.0:
                protected_attr_val, unique_label = 0.0, 0.0
            elif group == 3.0:
                protected_attr_val, unique_label = 0.0, 1.0
            elif group == 4.0:
                protected_attr_val, unique_label = 1.0, 0.0
            else:
                protected_attr_val, unique_label = 1.0, 1.0
                
            dataset_size = [len(metrics['member_signals']), len(metrics['non_member_signals'])]
            
            mia_result = MIA_Attack_Result(
                name="subpopulation_" + str(protected_attr_val) + "_label_"+str(unique_label),
                fpr=metrics['fprs'],
                tpr=metrics['tprs'],
                thresholds=metrics['thresholds'],
                auc_score=auc_score,
                privacy_risk=privacy_risk,
                accuracy=metrics['accuracy'],
                tpr_ind=metrics['tpr'],
                tnr_ind=metrics['tnr'],
                test_train_ratio=len(metrics['non_member_signals']) / len(metrics['member_signals']),
                dataset_size=dataset_size,
            )
            
            results.append(mia_result)
    
    print("========================================================================")
    
    return metric_results, pop_metric_results, group_metrics, pop_metrics, results
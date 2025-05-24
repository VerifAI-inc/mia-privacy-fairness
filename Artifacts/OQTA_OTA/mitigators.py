import numpy as np
import pandas as pd
from metrics_utils import compute_metrics, describe_metrics, get_test_metrics, test, get_test_metrics_for_syn_rew, get_test_metrics_for_eg, get_test_metrics_for_cpp

#setup test models
from models import TModel 

# Metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import utils

#Bias mitigation techniques
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import RejectOptionClassification

# Scalers
from sklearn.preprocessing import StandardScaler

from plot_utils import plot
import matplotlib as plt

#oversampling
from oversample import synthetic

from aif360.datasets import BinaryLabelDataset

from privacy_meter.dataset import Dataset

class BaseMitigator:

    def __init__(self):
        pass

    def run_mitigator(self):
        pass


# no mitigator
class NullMitigator(BaseMitigator): 

    mitigator_type = 'No Mitigator'
    def run_mitigator(self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test, 
                      model_type, orig_metrics, orig_mia_metrics, f_label, uf_label, 
                      unprivileged_groups, privileged_groups, ATTACK, THRESH_ARR, 
                      DISPLAY, SCALER, target_dataset, reference_dataset):
        dataset = dataset_orig_train
        metrics, mia_metrics = get_test_metrics(DATASET, target_dataset, reference_dataset, dataset, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, orig_mia_metrics, ATTACK, 'un_log', f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)
        return metrics, mia_metrics


class SyntheticMitigator(BaseMitigator):

    mitigator_type = 'Synthetic Data Mitigator'
    def run_mitigator (self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test, 
                       privileged_groups, unprivileged_groups, 
                       base_rate_privileged, base_rate_unprivileged, 
                       model_type, transf_metrics, transf_mia_metrics,
                       f_label, uf_label, 
                       ATTACK, THRESH_ARR, DISPLAY, OS_MODE, SCALER,
                       target_dataset, reference_dataset):
        
        # generating synthetic data
        dataset_transf_train = synthetic(dataset_orig_train, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, f_label, uf_label, os_mode = OS_MODE)
        print('origin, transf: ', dataset_orig_train.features.shape[0], dataset_transf_train.features.shape[0])

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        print('after transf priv: ', metric_transf_train.base_rate(privileged=True))
        print('after transf unpriv: ', metric_transf_train.base_rate(privileged=False))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


        # fitting the model on the transformed dataset with synthetic generator
        dataset = dataset_transf_train
        transf_metrics, transf_mia_metrics = get_test_metrics_for_syn_rew(DATASET, target_dataset, reference_dataset, dataset, dataset_orig_train, dataset_orig_val,
                                                                          dataset_orig_test, model_type, transf_metrics, transf_mia_metrics, ATTACK, 'syn_log', f_label,
                                                                          uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        return metric_transf_train, transf_metrics, transf_mia_metrics
    
def transform_privacy_meter_dataset(pm_dataset, DIR, feature_names, label_name, protected_attribute_name):
    print("=============================================================")
    print("TRANSFROM PRIVACY METER DATASET")
    transformed_data_dict = {}
    for split_name, split_data in pm_dataset.data_dict.items():
        x = split_data['x']
        y = split_data['y']
        g = split_data['g']

        # Create a DataFrame with features, labels, and sensitive attribute
        df = pd.DataFrame(x, columns=feature_names)
        df[label_name] = y

        # Extract sensitive attribute from x
        if protected_attribute_name in feature_names:
            sensitive_feature_index = feature_names.index(protected_attribute_name)
            sensitive_features = x[:, sensitive_feature_index]
            df[protected_attribute_name] = sensitive_features
        else:
            raise ValueError(f"Protected attribute '{protected_attribute_name}' not found in feature names.")
        
        print("DATAFRAME BEFORE DIR TRANSFORM", df)

        # Create a BinaryLabelDataset
        dataset = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=df,
            label_names=[label_name],
            protected_attribute_names=[protected_attribute_name],
            privileged_protected_attributes=[[1]],
            unprivileged_protected_attributes=[[0]]
        )

        # Apply DIR
        dataset_dir = DIR.fit_transform(dataset)

        # Extract transformed features
        x_transformed = dataset_dir.features
        y_transformed = dataset_dir.labels.ravel()

        # Update the split_data
        transformed_data_dict[split_name] = {
            'x': x_transformed,
            'y': y_transformed,
            'g': g
        }

    # Create a new privacy_meter Dataset
    transformed_pm_dataset = Dataset(
        data_dict=transformed_data_dict,
        default_input='x',
        default_output='y',
        default_group='g'
    )
    
    print("=====================================================================================")

    return transformed_pm_dataset

class DIRMitigator(BaseMitigator):

    mitigator_type = 'Disparity Impact Remover Mitigator'

    def run_mitigator(self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test, sensitive_attribute, model_type, dir_metrics, dir_mia_metrics,
                      f_label, uf_label, unprivileged_groups, privileged_groups, ATTACK, THRESH_ARR, DISPLAY, SCALER, target_dataset, reference_dataset):

        # Initialize the DisparateImpactRemover
        DIR = DisparateImpactRemover(sensitive_attribute=sensitive_attribute)

        # Apply DIR to the training, validation, and test datasets
        dataset_dir_train = DIR.fit_transform(dataset_orig_train)
        dataset_dir_val = DIR.fit_transform(dataset_orig_val)
        dataset_dir_test = DIR.fit_transform(dataset_orig_test)

        dataset = dataset_dir_train
        target_dataset_dir = None
        reference_dataset_dir = None

        # Apply DIR to target and reference datasets for MIA2 attack
        if ATTACK == 'mia2':
            # Extract feature names, label name, and protected attribute name from the original dataset
            feature_names = dataset_orig_train.feature_names
            label_name = dataset_orig_train.label_names[0]
            
            # Transform target_dataset and reference_dataset
            target_dataset_dir = transform_privacy_meter_dataset(pm_dataset=target_dataset, DIR=DIR, feature_names=feature_names, label_name=label_name,
                                                                 protected_attribute_name=sensitive_attribute)
            reference_dataset_dir = transform_privacy_meter_dataset(pm_dataset=reference_dataset, DIR=DIR, feature_names=feature_names, label_name=label_name,
                                                                    protected_attribute_name=sensitive_attribute)

        # Call get_test_metrics with transformed datasets
        dir_metrics, dir_mia_metrics = get_test_metrics( DATASET, target_dataset=target_dataset_dir, reference_dataset=reference_dataset_dir, dataset_orig_train=dataset,
                                                        dataset_orig_val=dataset_dir_val, dataset_orig_test=dataset_dir_test, model_type=model_type, 
                                                        test_metrics=dir_metrics, mia_metrics=dir_mia_metrics, ATTACK=ATTACK, log_type="dir_log", f_label=f_label, 
                                                        uf_label=uf_label, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, 
                                                        THRESH_ARR=THRESH_ARR, DISPLAY=DISPLAY, SCALER=SCALER)
        
        return dir_metrics, dir_mia_metrics

class ReweighMitigator(BaseMitigator):

    mitigator_type = 'Reweigh Mitigator'
    def run_mitigator(self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      f_label, uf_label, unprivileged_groups, privileged_groups, model_type, reweigh_metrics, reweigh_mia_metrics,
                      ATTACK, THRESH_ARR, DISPLAY, SCALER, target_dataset, reference_dataset):

        # transform the data with preprocessing reweighing and fit the model
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_reweigh_train = RW.fit_transform(dataset_orig_train)

        dataset = dataset_reweigh_train
        
        reweigh_metrics, reweigh_mia_metrics = get_test_metrics_for_syn_rew(DATASET, target_dataset, reference_dataset, dataset, dataset_orig_train, dataset_orig_val, 
                                                                            dataset_orig_test, model_type, reweigh_metrics, reweigh_mia_metrics, ATTACK, 'rew_log', 
                                                                            f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        
        return reweigh_metrics, reweigh_mia_metrics


# Exponentiated Gradiant Reduction (In-processing)
class EGMitigator(BaseMitigator):
    mitigator_type = 'EG Mitigator'
    def run_mitigator(self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      eg_metrics, eg_mia_metrics, model_type, 
                      f_label, uf_label, 
                      unprivileged_groups, privileged_groups, 
                      ATTACK, THRESH_ARR, DISPLAY, SCALER, target_dataset, reference_dataset):
        
        dataset = dataset_orig_train
        eg_metrics, eg_mia_metrics = get_test_metrics_for_eg(DATASET, target_dataset, reference_dataset, dataset, dataset_orig_val, dataset_orig_test, model_type, 
                                                             eg_metrics, eg_mia_metrics, ATTACK, 'eg_log', f_label, uf_label, unprivileged_groups, privileged_groups, 
                                                             THRESH_ARR, DISPLAY, SCALER)
                
        return eg_metrics, eg_mia_metrics

# Calibrated Equalized Odds (Post-Processing)
class CPPMitigator(BaseMitigator):
    mitigator_type = 'CPP Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, f_label, uf_label,  unprivileged_groups, privileged_groups, model_type, 
                      cpp_metrics, cpp_mia_metrics, ATTACK, THRESH_ARR, DISPLAY, SCALER, target_dataset=None, reference_dataset=None):
        # set up dataset
        dataset = dataset_orig_train
        cpp_metrics, cpp_mia_metrics = get_test_metrics_for_cpp(target_dataset, reference_dataset, dataset, dataset_orig_val, dataset_orig_test, model_type, 
                                                                cpp_metrics, cpp_mia_metrics, ATTACK, 'cpp_log', f_label, uf_label, unprivileged_groups, 
                                                                privileged_groups, THRESH_ARR, DISPLAY, SCALER)
                                
        return cpp_metrics, cpp_mia_metrics
    
class SyntheticEGMitigator(BaseMitigator):

    mitigator_type = 'Synthetic and EG Mitigator'
    def run_mitigator (self, DATASET, dataset_orig_train, dataset_orig_val, dataset_orig_test, 
                       privileged_groups, unprivileged_groups, 
                       base_rate_privileged, base_rate_unprivileged, 
                       model_type, syn_eg_metrics, syn_eg_mia_metrics,
                       f_label, uf_label, 
                       ATTACK, THRESH_ARR, DISPLAY, OS_MODE, SCALER,
                       target_dataset, reference_dataset):
        
        # generating synthetic data
        dataset_transf_train = synthetic(dataset_orig_train, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, f_label, uf_label, os_mode = OS_MODE)
        print('origin, transf: ', dataset_orig_train.features.shape[0], dataset_transf_train.features.shape[0])

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        print('after transf priv: ', metric_transf_train.base_rate(privileged=True))
        print('after transf unpriv: ', metric_transf_train.base_rate(privileged=False))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


        # fitting the model on the transformed dataset with synthetic generator
        dataset = dataset_transf_train
        
        syn_eg_metrics, syn_eg_mia_metrics = get_test_metrics_for_eg(DATASET, target_dataset, reference_dataset, dataset, dataset_orig_val, dataset_orig_test, model_type, 
                                                             syn_eg_metrics, syn_eg_mia_metrics, ATTACK, 'syn_eg_log', f_label, uf_label, unprivileged_groups, privileged_groups, 
                                                             THRESH_ARR, DISPLAY, SCALER)
     
        return syn_eg_metrics, syn_eg_mia_metrics
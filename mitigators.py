import numpy as np
import pandas as pd
from metrics_utils import compute_metrics, describe_metrics, get_test_metrics, test, get_test_metrics_for_syn

#setup test models
from models import TModel, MLPClassifierWrapper 

# Metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import utils

#Bias mitigation techniques
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, OptimPreproc, Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, ARTClassifier, GerryFairClassifier, MetaFairClassifier, PrejudiceRemover
# from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing\
        import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing\
        import EqOddsPostprocessing
# from aif360.algorithms.postprocessing.reject_option_classification\
#        import RejectOptionClassification
from aif360.algorithms.postprocessing import RejectOptionClassification

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from explainer import Explainer

from plot_utils import plot
import matplotlib as plt

#oversampling
from oversample import synthetic

from membership_infer_attack import run_mia_attack

from collections import OrderedDict, defaultdict

class BaseMitigator:

    def __init__(self):
        pass

    def run_mitigator(self):
        pass


# no mitigator
class NullMitigator(BaseMitigator): 

    mitigator_type = 'No Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, 
                      model_type, orig_metrics, orig_mia_metrics,
                      f_label, uf_label, 
                      unprivileged_groups, privileged_groups, 
                      THRESH_ARR, DISPLAY, SCALER):
        dataset = dataset_orig_train
        metrics, mia_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, orig_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)
    
        # For exp
        # explainer = Explainer()
        # explainer.tree_explain(dataset, dataset_orig_test, unprivileged_groups, 'orig')
        return metrics, mia_metrics


class SyntheticMitigator(BaseMitigator):

    mitigator_type = 'Synthetic Data Mitigator'
    def run_mitigator (self, dataset_orig_train, dataset_orig_val, dataset_orig_test, 
                       privileged_groups, unprivileged_groups, 
                       base_rate_privileged, base_rate_unprivileged, 
                       model_type, transf_metrics, transf_mia_metrics,
                       f_label, uf_label, 
                       THRESH_ARR, DISPLAY, OS_MODE, SCALER):
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
        # transf_metrics, transf_mia_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, transf_metrics, transf_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)
        
        transf_metrics, transf_mia_metrics = get_test_metrics_for_syn(dataset, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, transf_metrics, transf_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        # For exp
        #explainer = Explainer()
        #explainer.tree_explain(dataset_transf_train, dataset_orig_test, unprivileged_groups, 'synth')

        return metric_transf_train, transf_metrics, transf_mia_metrics


class DIRMitigator(BaseMitigator):

    mitigator_type = 'Disparity Impact Remover Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,  
                      sensitive_attribute, model_type, dir_metrics, dir_mia_metrics,
                      f_label, uf_label, unprivileged_groups, privileged_groups, 
                      THRESH_ARR, DISPLAY, SCALER):
        DIR = DisparateImpactRemover(sensitive_attribute=sensitive_attribute)
        dataset_dir_train = DIR.fit_transform(dataset_orig_train)
        dataset_dir_val = DIR.fit_transform(dataset_orig_val)
        dataset_dir_test = DIR.fit_transform(dataset_orig_test)

        dataset = dataset_dir_train
        # For exp
        #explainer = Explainer()
        #explainer.explain(dataset, dataset_orig_test, 'dir')

        dir_metrics, dir_mia_metrics = get_test_metrics(dataset, dataset_dir_val, dataset_dir_test, model_type, dir_metrics, dir_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        return dir_metrics, dir_mia_metrics


class OPMitigator(BaseMitigator):

    mitigator_type = 'Optim Prepproc Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,  
                      sensitive_attribute, model_type, op_metrics, SCALER):
        OP = OptimPreproc(unprivileged_groups=unprivileged_groups,
                          privileged_groups=privileged_groups)
        dataset_op_train = OP.fit_transform(dataset_orig_train)

        return op_metrics

class ReweighMitigator(BaseMitigator):

    mitigator_type = 'Reweigh Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      f_label, uf_label, unprivileged_groups, privileged_groups, model_type, reweigh_metrics, reweigh_mia_metrics,
                      THRESH_ARR, DISPLAY, SCALER):

        # transform the data with preprocessing reweighing and fit the model
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_reweigh_train = RW.fit_transform(dataset_orig_train)

        dataset = dataset_reweigh_train
        
        """

        test_model = TModel(model_type)
        mod_reweigh = test_model.set_model(dataset, SCALER)

        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
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
                           model=mod_reweigh,
                           thresh_arr=thresh_arr, metric_arrs=None)
        reweigh_best_ind = np.argmax(val_metrics['bal_acc'])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, model_type + ' Reweighing Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, model_type + ' Reweighing Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 val_metrics['avg_odds_diff'], 'avg. odds diff.')
            plt.show()

        reweigh_metrics = test(f_label, uf_label,
                               unprivileged_groups, privileged_groups,
                               dataset=dataset_orig_test_pred,
                               model=mod_reweigh,
                               thresh_arr=[thresh_arr[reweigh_best_ind]], metric_arrs=reweigh_metrics)

        describe_metrics(reweigh_metrics, thresh_arr)

        explainer = Explainer()
        explainer.explain(dataset_reweigh_train, dataset_orig_test, 'reweigh')
        """
        # For exp

        
        #explainer = Explainer()
        #explainer.explain(dataset_reweigh_train, dataset_orig_test, 'reweigh')
        
        reweigh_metrics, reweigh_mia_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, reweigh_metrics, reweigh_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        
        return reweigh_metrics, reweigh_mia_metrics


# Exponentiated Gradiant Reduction (In-processing)
class EGRMitigator(BaseMitigator):
    mitigator_type = 'EGR Mitigator'
 
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      egr_metrics, egr_mia_metrics, model_type, 
                      f_label, uf_label, 
                      unprivileged_groups, privileged_groups, 
                      THRESH_ARR, DISPLAY, SCALER):
        # set up dataset
        if SCALER:
            pr_orig_scaler = StandardScaler()
            dataset = dataset_orig_train.copy(deepcopy=True)
            dataset.features = pr_orig_scaler.fit_transform(dataset.features)

            dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
            dataset_val_pred.features = pr_orig_scaler.transform(dataset_val_pred.features)

            dataset_test_pred = dataset_orig_test.copy(deepcopy=True)
            dataset_test_pred.features = pr_orig_scaler.transform(dataset_test_pred.features)

        else:
            dataset = dataset_orig_train.copy(deepcopy=True)
            dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
            dataset_test_pred = dataset_orig_test.copy(deepcopy=True)

        dataset.labels = dataset.labels.ravel()
        test_model = TModel(model_type)
        model = test_model.get_model()
        print('fitting EGR ...')
        #np.random.seed(0) #need for reproducibility
        egr_mod = ExponentiatedGradientReduction(estimator=model,
                                                 prot_attr=['age'],
                                                #  T=2,
                                                eta0=2.0,
                                                eps=0.01,
                                                constraints="EqualizedOdds",
                                                drop_prot_attr=False)

        # After applying EGR
        # Convert dataset.features (NumPy array) to a DataFrame
        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        y_series = pd.Series(dataset.labels.ravel())
        egr_mod.fit(X_df, y_series)

        #validate
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

        val_metrics = test(f_label, uf_label,
                           unprivileged_groups, privileged_groups,
                           dataset=dataset_val_pred,
                           model=egr_mod,
                           thresh_arr=thresh_arr, metric_arrs=None)
        
        egr_best_ind = np.argmax(val_metrics['bal_acc'])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, 'Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, 'Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 val_metrics['avg_odds_diff'], 'avg. odds diff.')

            plt.show()

        egr_metrics = test(f_label, uf_label,
                           unprivileged_groups, privileged_groups,
                           dataset=dataset_test_pred,
                           model=egr_mod,
                           thresh_arr=[thresh_arr[egr_best_ind]], metric_arrs=egr_metrics)

        describe_metrics(egr_metrics, [thresh_arr[egr_best_ind]])
        
        # Runnning MIA attack based on subgroups
        results = run_mia_attack(privileged_groups, dataset_orig_train, dataset_orig_test, model_type, egr_mod)
            
        for i in results:
            print(i)
            
        # metrics array to hold the results
        if egr_mia_metrics is None:
            egr_mia_metrics = defaultdict(list)

        # Add the results to test_metrics object
        # MIA results for overall dataset and subpopulations
        for i in range(len(results)):
            egr_mia_metrics[f"{results[i].get_name()}_mia_auc"].append(results[i].get_auc())
            egr_mia_metrics[f"{results[i].get_name()}_mia_privacy_risk"].append(results[i].get_privacy_risk())
            egr_mia_metrics[f"{results[i].get_name()}_mia_ppv"].append(results[i].get_ppv())
            egr_mia_metrics[f"{results[i].get_name()}_mia_attacker_advantage"].append(results[i].get_attacker_advantage())
            egr_mia_metrics[f"{results[i].get_name()}_mia_result"].append(results[i])
        
        #exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)
        return egr_metrics, egr_mia_metrics


# Prejudice Remover (Post-processing)
class PRMitigator(BaseMitigator):

    mitigator_type = 'PR Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      pr_orig_metrics, sens_attr, 
                      f_label, uf_label, 
                      unprivileged_groups, privileged_groups, 
                      THRESH_ARR, DISPLAY, SCALER):
        SCALER = True

        # train 
        model = PrejudiceRemover(sensitive_attr=sens_attr, eta=2.0)
        if SCALER:
            pr_orig_scaler = StandardScaler()
            dataset = dataset_orig_train.copy(deepcopy=True)
            dataset.features = pr_orig_scaler.fit_transform(dataset.features)

            dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
            dataset_val_pred.features = pr_orig_scaler.transform(dataset_val_pred.features)

            dataset_test_pred = dataset_orig_test.copy(deepcopy=True)
            dataset_test_pred.features = pr_orig_scaler.transform(dataset_test_pred.features)

        else:
            dataset = dataset_orig_train.copy(deepcopy=True)
            dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
            dataset_test_pred = dataset_orig_test.copy(deepcopy=True)

        pr_orig = model.fit(dataset)

        #validate 
        thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

        val_metrics = test(f_label, uf_label,
                           unprivileged_groups, privileged_groups,
                           dataset=dataset_val_pred,
                           model=pr_orig,
                           thresh_arr=thresh_arr, metric_arrs=None)
        pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])

        disp_imp = np.array(val_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

        if DISPLAY:
            plot(thresh_arr, 'Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 disp_imp_err, '1 - min(DI, 1/DI)')

            plot(thresh_arr, 'Classification Thresholds',
                 val_metrics['bal_acc'], 'Balanced Accuracy',
                 val_metrics['avg_odds_diff'], 'avg. odds diff.')

            plt.show()

        pr_orig_metrics = test(f_label, uf_label,
                               unprivileged_groups, privileged_groups,
                               dataset=dataset_test_pred,
                               model=pr_orig,
                               thresh_arr=[thresh_arr[pr_orig_best_ind]], metric_arrs=pr_orig_metrics)

        describe_metrics(pr_orig_metrics, [thresh_arr[pr_orig_best_ind]])

        return pr_orig_metrics


# Cost Constrained (Post-processing)
class CPPMitigator(BaseMitigator):

    mitigator_type = 'CPP Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      cpp_orig_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER):
        # cost constraint of fnr will optimize generalized false negative rates, that of
        # fpr will optimize generalized false positive rates, and weighted will optimize
        # a weighted combination of both
        cost_constraint = "weighted" # "fnr", "fpr", "weighted"

        #random seed for calibrated equal odds prediction
        np.random.seed(1)

        # Verify metric name
        allowed_constraints = ["fnr", "fpr", "weighted"]
        if cost_constraint not in allowed_constraints:
            raise ValueError("Constraint should be one of allowed constraints")

        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

        if SCALER:
            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)

            dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
            X_valid = scale_orig.transform(dataset_orig_valid_pred.features)

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            X_test = scale_orig.transform(dataset_orig_test_pred.features)

        else:
            X_train = dataset_orig_train.features

            dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
            X_valid = dataset_orig_valid_pred.features

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            X_test = dataset_orig_test_pred.features

        y_train = dataset_orig_train.labels.ravel()

        # Logistic regression classifier and predictions
        test_model = TModel(model_type)
        lmod = test_model.get_model()
        lmod.fit(X_train, y_train)
        y_train_pred = lmod.predict(X_train)

        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

        dataset_orig_train_pred.labels = y_train_pred
        y_valid = dataset_orig_valid_pred.labels.ravel()
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

        y_test = dataset_orig_test_pred.labels.ravel()
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

        num_thresh = 50
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, THRESH_ARR, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):

            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

            classified_metric_orig_valid = ClassificationMetric(dataset_orig_val,
                                                     dataset_orig_valid_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

            ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                               +classified_metric_orig_valid.true_negative_rate())

        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]

        # Learn parameters to equalize odds and apply to create a new dataset
        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                             unprivileged_groups = unprivileged_groups,
                                             cost_constraint=cost_constraint,
                                             seed=None)
        cpp = cpp.fit(dataset_orig_val, dataset_orig_valid_pred)
        # Metrics for the validation set
        fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        #print("#### Validation set")
        #print("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

        metric_valid_bef = compute_metrics(dataset_orig_val, dataset_orig_valid_pred,
                           unprivileged_groups, privileged_groups, None, disp=False)

        # Transform the validation set
        #dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)

        #print("#### Validation set")
        #print("##### Transformed predictions - With fairness constraints")
        metric_valid_aft = compute_metrics(dataset_orig_val, dataset_transf_valid_pred,
                           unprivileged_groups, privileged_groups, None, disp=False)
        #print(metric_valid_aft)

        # Testing: Check if the metric optimized has not become worse
        #assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])

        # Metrics for the test set
        fav_inds = dataset_orig_test_pred.scores > best_class_thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                          unprivileged_groups, privileged_groups, None, disp = False)

        #print(metric_test_bef)

        # Metrics for the transformed test set
        #dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

        #print("#### Test set")
        #print("##### Transformed predictions - With fairness constraints")
        cpp_orig_metrics = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                          unprivileged_groups, privileged_groups, cpp_orig_metrics, disp=False)

        describe_metrics(cpp_orig_metrics, [best_class_thresh]) #[thresh_arr[pr_orig_best_ind]])

        return cpp_orig_metrics



# Reject Option (Post-processing)
class ROMitigator(BaseMitigator):

    mitigator_type = 'RO Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      ro_orig_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER):
        # Metric used (should be one of allowed_metrics)
        metric_name = "Statistical parity difference"

        # Upper and lower bound on the fairness metric used
        metric_ub = 0.05
        metric_lb = -0.05

        #random seed for calibrated equal odds prediction
        #np.random.seed(1)

        # Verify metric name
        allowed_metrics = ["Statistical parity difference",
                           "Average odds difference",
                           "Equal opportunity difference"]
        if metric_name not in allowed_metrics:
            raise ValueError("Metric name should be one of allowed metrics")

        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

        if SCALER:
            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)

            dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
            X_valid = scale_orig.transform(dataset_orig_valid_pred.features)

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            X_test = scale_orig.transform(dataset_orig_test_pred.features)

        else:
            X_train = dataset_orig_train.features

            dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
            X_valid = dataset_orig_valid_pred.features

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            X_test = dataset_orig_test_pred.features

        y_train = dataset_orig_train.labels.ravel()

        # Logistic regression classifier and predictions
        test_model = TModel(model_type)
        lmod = test_model.get_model()
        lmod.fit(X_train, y_train)
        y_train_pred = lmod.predict(X_train)

        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

        dataset_orig_train_pred.labels = y_train_pred
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

        num_thresh = 50
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, THRESH_ARR, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):

            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

            classified_metric_orig_valid = ClassificationMetric(dataset_orig_val,
                                                     dataset_orig_valid_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

            ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                               +classified_metric_orig_valid.true_negative_rate())

        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]

        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups,
                                         low_class_thresh=0.01, high_class_thresh=THRESH_ARR,
                                          num_class_thresh=100, num_ROC_margin=50,
                                          metric_name=metric_name,
                                          metric_ub=metric_ub, metric_lb=metric_lb)
        ROC = ROC.fit(dataset_orig_val, dataset_orig_valid_pred)

        # Metrics for the validation set
        fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        #print("#### Validation set")
        #print("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

        metric_valid_bef = compute_metrics(dataset_orig_val, dataset_orig_valid_pred,
                           unprivileged_groups, privileged_groups, None, disp=False)

        # Transform the validation set
        dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

        #print("#### Validation set")
        #print("##### Transformed predictions - With fairness constraints")
        metric_valid_aft = compute_metrics(dataset_orig_val, dataset_transf_valid_pred,
                           unprivileged_groups, privileged_groups, None, disp=False)
        #print(metric_valid_aft)

        # Testing: Check if the metric optimized has not become worse
        #assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])

        # Metrics for the test set
        fav_inds = dataset_orig_test_pred.scores > best_class_thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                          unprivileged_groups, privileged_groups, None, disp = False)

        #print(metric_test_bef)

        # Metrics for the transformed test set
        dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

        #print("#### Test set")
        #print("##### Transformed predictions - With fairness constraints")
        ro_orig_metrics = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                          unprivileged_groups, privileged_groups, ro_orig_metrics, disp=False)

        describe_metrics(ro_orig_metrics, [best_class_thresh]) #[thresh_arr[pr_orig_best_ind]])

        return ro_orig_metrics
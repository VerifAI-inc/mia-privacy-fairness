from mitigators import BaseMitigator, NullMitigator, SyntheticMitigator, DIRMitigator, ReweighMitigator, EGRMitigator, PRMitigator, CPPMitigator, ROMitigator

class TestAlgorithms:

    def __init__(self, model_type):
        self.model_type = model_type
        print('#######################################################################')
        print('                    '+ self.model_type)
        print('#######################################################################')

    def run_original(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, orig_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
        print('\n------------------------------\n')
        print('[INFO] Original Results......')
        print('\n------------------------------\n')

        null_mitigator = NullMitigator()
        orig_metrics, orig_mia_metrics = null_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, orig_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        # null_mitigator.run_explainer(dataset_orig_train, dataset_orig_test, model_type, SCALER)
        return orig_metrics, orig_mia_metrics


    def run_oversample(self,dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, transf_mia_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER):
        print('\n------------------------------\n')
        print('[INFO] Random Oversampling ......')
        print('\n------------------------------\n')
        synth_mitigator = SyntheticMitigator()
        metric_transf_train, transf_metrics, transf_mia_metrics = synth_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, transf_mia_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER)

        #synth_mitigator.run_explainer(dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER)
        return metric_transf_train, transf_metrics, transf_mia_metrics 

    def run_dir(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,  sens_attr, model_type, dir_metrics, dir_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
        print('\n------------------------------\n')
        print('[INFO] preprocessing--disparat impact remover ......')
        print('\n------------------------------\n')
        dir_mitigator = DIRMitigator()
        dir_metrics, dir_mia_metrics = dir_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  sens_attr, model_type, dir_metrics, dir_mia_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        return dir_metrics, dir_mia_metrics

    def run_rew(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, f_label, uf_label,  unprivileged_groups, privileged_groups, model_type, reweigh_metrics, reweigh_mia_metrics, THRESH_ARR, DISPLAY, SCALER):
        print('\n------------------------------\n')
        print('[INFO] preprocessing--reweighting ......')
        print('\n------------------------------\n')
        rew_mitigator = ReweighMitigator()
        reweigh_metrics, reweigh_mia_metrics = rew_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, f_label, uf_label,  unprivileged_groups, privileged_groups, model_type, reweigh_metrics, reweigh_mia_metrics, THRESH_ARR, DISPLAY, SCALER)

        return reweigh_metrics, reweigh_mia_metrics

    def run_egr(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, egr_metrics, model_type, f_label, uf_label, unprivileged_groups, privileged_groups,THRESH_ARR, DISPLAY, SCALER):
        print('\n------------------------------\n')
        print('\n[INFO:] in-processing Exponentiation Gradient Reduction ...... \n')
        print('\n------------------------------\n')
        egr_mitigator = EGRMitigator()
        egr_metrics = egr_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, egr_metrics, model_type, f_label, uf_label, unprivileged_groups, privileged_groups,THRESH_ARR, DISPLAY, SCALER)

        return egr_metrics

    def run_pr(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, pr_orig_metrics, sens_attr, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
        if self.model_type == 'lr':
            print('\n------------------------------\n')
            print('\n[INFO:] in-processing Prejudice Remover ...... \n')
            print('\n------------------------------\n')
            pr_mitigator = PRMitigator()
            pr_orig_metrics = pr_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, pr_orig_metrics, sens_attr, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

            return pr_orig_metrics

    def run_cpp(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, cpp_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER):
        print('\n------------------------------\n')
        print('\n[INFO:] post-processing Calibrated Equal Odds ......\n')
        print('\n------------------------------\n')
        cpp_mitigator = CPPMitigator()
        cpp_metrics = cpp_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, cpp_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER)

        return cpp_metrics

    def run_ro(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, ro_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER):
        print('\n------------------------------\n')
        print('\n[INFO:] post-processing Reject Option Classification ......\n')
        print('\n------------------------------\n')
        ro_mitigator = ROMitigator()
        ro_metrics = ro_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, ro_metrics, model_type, unprivileged_groups, privileged_groups, THRESH_ARR, SCALER)

        return ro_metrics

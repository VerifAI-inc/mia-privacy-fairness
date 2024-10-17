import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from oversample import group_indices




class Explainer():
    def __init__(self):
        pass

    def explain (self, dataset_orig_train, dataset_orig_test, mi_type):

        model = LogisticRegression(penalty="l2", C=0.1)
        model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        explainer = shap.LinearExplainer(model, dataset_orig_train.features, feature_dependence="independent")
        shap_values = explainer.shap_values(dataset_orig_test.features)
        X_test_array = dataset_orig_test.features # we need to pass a dense version for the plotting functions
        shap.summary_plot(shap_values, X_test_array, dataset_orig_train.feature_names)
        #shap.plots.scatter(shap_values[:,"race"], color=shap_values)
        plt.show()
        # plt.savefig('./eps/summary_plot_'+mi_type+'.jpg')


    def tree_explain (self, dataset_orig_train, dataset_orig_test, unprivileged_groups, mi_type):

        # make a duplicate copy of the input data
        dataset = dataset_orig_test

        # indices of examples in the unprivileged and privileged groups
        indices, priv_indices = group_indices (dataset, unprivileged_groups)

        # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset
        unprivileged_dataset = dataset.subset(indices) # unprivileaged
        privileged_dataset = dataset.subset(priv_indices) # privilegaed

        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
        model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dataset_orig_test.features)
        X_test_array = dataset_orig_test.features # we need to pass a dense version for the plotting functions
        #shap_values = explainer.shap_values(unprivileged_dataset.features)
        #X_test_array = unprivileged_dataset.features # we need to pass a dense version for the plotting functions
        shap.summary_plot(shap_values, X_test_array, dataset_orig_train.feature_names)
        plt.show()
        # plt.savefig('./eps/summary_tree_plot_'+mi_type+'.jpg')
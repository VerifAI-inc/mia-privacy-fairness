import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from oversample import group_indices


class Explainer:
    def explain(self, dataset_orig_train, dataset_orig_test, mi_type):
        """
        Generate SHAP explanation for a logistic regression model.
        """
        model = LogisticRegression(penalty="l2", C=0.1)
        model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        explainer = shap.LinearExplainer(
            model,
            dataset_orig_train.features,
            feature_dependence="independent"
        )
        shap_values = explainer.shap_values(dataset_orig_test.features)

        X_test_array = dataset_orig_test.features
        shap.summary_plot(shap_values, X_test_array, dataset_orig_train.feature_names)
        plt.show()
        # plt.savefig('./eps/summary_plot_' + mi_type + '.jpg')

    def tree_explain(self, dataset_orig_train, dataset_orig_test, unprivileged_groups, mi_type):
        """
        Generate SHAP explanation for a Random Forest model on the full test set.
        """
        # Identify unprivileged and privileged group indices (not used in plot but available for future filtering)
        indices, priv_indices = group_indices(dataset_orig_test, unprivileged_groups)
        _ = dataset_orig_test.subset(indices)      # unprivileged_dataset
        _ = dataset_orig_test.subset(priv_indices) # privileged_dataset

        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
        model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dataset_orig_test.features)

        X_test_array = dataset_orig_test.features
        shap.summary_plot(shap_values, X_test_array, dataset_orig_train.feature_names)
        plt.show()
        # plt.savefig('./eps/summary_tree_plot_' + mi_type + '.jpg')
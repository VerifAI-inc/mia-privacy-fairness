import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
from typing import Union, List
from dataclasses import dataclass
    
_ABSOLUTE_TOLERANCE = 1e-3
@dataclass
class MIA_Attack_Result:

    name:str

    # False positive rates based on thresholds
    fpr: np.ndarray

    # True positive rates based on thresholds
    tpr: np.ndarray

    # Thresholds used to define points on ROC curve.
    # Thresholds are not explicitly part of the curve, and are stored for
    # debugging purposes.
    thresholds: np.ndarray

    #auc score
    auc_score:np.float64
        
    # privacy risk
    # In Hongyan Chang and Reza Shokri, "On the Privacy Risks of Algorithmic Fairness"
    privacy_risk : np.float64

    # overall_accuracy
    accuracy : np.float64

    
    #train acc and test accuracy
    tpr_ind : np.float64
    
    tnr_ind : np.float64



    # Ratio of test to train set size.
    # In Jayaraman et al. (https://arxiv.org/pdf/2005.10881.pdf) it is referred to
    # as 'gamma' (see Table 1 for the definition).
    test_train_ratio: np.float64

    dataset_size: List[int]

    def get_name(self):
        """Returns the name of the population"""
        return self.name
    
    def get_privacy_risk(self):
        """Returns the privacy_risk of the population"""
        return self.privacy_risk

    def get_auc(self):
        """Calculates area under curve (aka AUC)."""
        return metrics.auc(self.fpr, self.tpr)

    def get_attacker_advantage(self):
        """Calculates membership attacker's (or adversary's) advantage.

        This metric is inspired by https://arxiv.org/abs/1709.01604, specifically
        by Definition 4. The difference here is that we calculate maximum advantage
        over all available classifier thresholds.

        Returns:
          a single float number with membership attacker's advantage.
        """
        return max(np.abs(self.tpr - self.fpr))

    def get_ppv(self) -> float:
        # The Positive Predictive Value (PPV) is the proportion of positive
        # predictions that are true positives. It is expressed as PPV=TP/(TP+FP).
        # It was suggested by Jayaraman et al.
        # (https://arxiv.org/pdf/2005.10881.pdf) that this would be a suitable
        # metric for membership attack models trained on datasets where the number
        # of samples from the training set and the number of samples from the test
        # set are very different. These are referred to as imbalanced datasets.
        num = np.asarray(self.tpr)
        den = num + np.asarray([r * self.test_train_ratio for r in self.fpr])

        # Find when `tpr` and `fpr` are 0.
        tpr_is_0 = np.isclose(self.tpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
        fpr_is_0 = np.isclose(self.fpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
        tpr_and_fpr_both_0 = np.logical_and(tpr_is_0, fpr_is_0)
        # PPV when both are zero is given by the expression below.
        ppv_when_tpr_fpr_both_0 = 1. / (1. + self.test_train_ratio)
        # PPV when one is not zero is given by the expression below.
        ppv_when_one_of_tpr_fpr_not_0 = np.divide(
            num, den, out=np.zeros_like(den), where=den != 0)
        return np.max(
            np.where(tpr_and_fpr_both_0, ppv_when_tpr_fpr_both_0,
                     ppv_when_one_of_tpr_fpr_not_0))

    def __str__(self):
        """Returns AUC, advantage and PPV metrics."""
        optimal_idx = np.argmax(self.tpr - self.fpr)
        optimal_threshold = self.thresholds[optimal_idx]
        return '\n'.join([
            'MIA_Result(',
            '  Name: ' + self.name,
            f"  Size of the Dataset: Train = {self.dataset_size[0]}, Test = {self.dataset_size[1]}",
            '  AUC: %.2f' % self.get_auc(),
            '  Privacy Risk: %.2f' % self.privacy_risk,
            '  Accuracy: %.2f' % self.accuracy,
            '  Train Accuracy (TPR): %.2f' % self.tpr_ind,
            '  Test Accuracy (TNR): %.2f' % self.tnr_ind,
            '  Attacker advantage: %.2f' % self.get_attacker_advantage(),
            '  Positive predictive value: %.2f' % self.get_ppv(),
            '  Optimal thershold: ' + str(optimal_threshold),
            ')'
        ])


    def display_roc_curve(self):
        "Displaying the curve"
        #roc_auc = metrics.auc(self.fpr, self.tpr)
        #display = metrics.RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=roc_auc)
        #display.plot()
        #plt.show()

        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k', lw=1.0)
        plt.plot(self.fpr, self.tpr, lw=2, label=f'AUC: {metrics.auc(self.fpr, self.tpr):.3f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        #return fig
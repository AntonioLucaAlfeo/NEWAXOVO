from time import time
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y


class ExplanationsUtils:

    """
      This class implements some useful functions for obtaining explanations for given samples.
    """

    @staticmethod
    def random_forest_explanations(x, y, n_classes):

        """
          This function creates a new RandomForestClassifier and generates explanations from that model for the given
          data. Explanations for each class are computed as the sum of explanations of all samples that are part of
          that class.

          | Pre-conditions: none.
          | Post-conditions: RandomForestExplanation for the given data are computed.
          | Main output: explanations for each class.

        :param x: an array that contains features of the samples for which explanation is required.
        :type x: ndarray (n_samples, n_features).
        :param y: an array containing labels for the samples in *x*.
        :type y: ndarray (n_samples,)
        :param n_classes: number of classes of the considered problem.
        :type n_classes: unsigned int.
        :return: explanation for the given classes.
        :rtype: ndarray (n_classes, n_features).
        :raise: none.
        """

        x, y = check_X_y(x, y)

        explanations = np.zeros((n_classes, x.shape[1]))

        tmp_model = RandomForestClassifier().fit(x, y)

        explainer = shap.TreeExplainer(tmp_model)

        for class_index in np.arange(n_classes):

            relevant_rows = np.where(y == class_index + 1)[0]
            e_x = np.array(x[relevant_rows, :])
            shap_values = explainer.shap_values(e_x)

            explanations[class_index, :] += shap_values[class_index].sum(axis=0)

        return explanations



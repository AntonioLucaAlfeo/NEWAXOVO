import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from joblib import dump
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from src.Utils.DatasetUtils import DatasetUtils
from src.Utils.ClassifiersManager import ClassifiersManager


class OVOBaseLayer:

    """
      This class implements the base layer for the TAWAXOVO architecture. Base classifiers are trained in a 1 vs 1
      manner.
    """

    def __init__(self, n_classes, subject_id, models_path, results_path, samples_balancing):

        """
          This function initializes the OVOBaseLayer (TAWAX version) object.

          | Pre-conditions: none.
          | Post-conditions: a new OVOBaseLayer (TAWAX version) object is created.
          | Main output: none.

        :param n_classes: number of classes of the given problem.
        :type n_classes: int (>0).
        :param subject_id: id of the subject considered.
        :type subject_id: int (between 1 and 9).
        :param models_path: path to directory in which models are saved.
        :type models_path: str.
        :param results_path: path to directory in which models are saved.
        :type results_path: str.
        :param samples_balancing: balancing strategy adopted in case of non-balanced sub-problem. One between 'none',
            'random', 'clustering', 'smote', 'nearmiss'.
        :type samples_balancing: str.
        :return: none.
        :raise: none.
        """

        self.results_path = results_path
        self.models_path = models_path
        self.n_classes = n_classes
        self.subject_id = subject_id
        self.samples_balancing = samples_balancing

        self.n_base_classifiers = int(n_classes * (n_classes - 1) / 2)
        self.base_classifiers_dictionary = np.array([[i, j] for i in np.arange(start=1, stop=self.n_classes + 1)
                                                     for j in np.arange(start=i + 1, stop=self.n_classes + 1)])

        # NON CLASS-DEPENDENT CLASSIFIERS
        classifiers_configuration_name = 'optuna_500_subject_%d_classes_%d' % (subject_id, n_classes)

        # UCI DATASETS
        # classifiers_configuration_name = 'mlp'

        self.base_classifiers = [ClassifiersManager.return_classifier(classifiers_configuration_name)
                                 for _ in np.arange(self.n_base_classifiers)]

        self.training_scores = np.zeros(self.n_base_classifiers)

    def fit(self, x, y):

        """
          This function fits the 1 vs 1 classifiers of the base layer on the given training set.

          | Pre-conditions: :py:meth:'OVOBaseLayer.__init__'.
          | Post-conditions: the OVOBaseLayer object is trained on the given data.
          | Main output: none.

        :param x: a matrix representing the features of each sample on which the layer is trained.
        :type x: ndarray (n_samples*n_features).
        :param y: an array of labels corresponding to the given features.
        :type y: ndarray (n_samples,).
        :return: self.
        :rtype: estimator.
        :raise: none.
        """

        x, y = check_X_y(x, y)

        for classifier_index, specialist_classes in enumerate(self.base_classifiers_dictionary):

            relevant_rows = np.where(np.logical_or(y == specialist_classes[0], y == specialist_classes[1]))[0]
            training_x = np.array(x[relevant_rows, :])
            training_y = np.where(y[relevant_rows] == specialist_classes[0], 1, 0)

            if len(np.where(training_y == 0)[0]) != len(np.where(training_y == 1)[0]):

                if self.samples_balancing == "random":

                    training_x, training_y = RandomUnderSampler(sampling_strategy=1.0).fit_resample(training_x, training_y)

                elif self.samples_balancing == "clustering":

                    training_x, training_y = ClusterCentroids(sampling_strategy=1.0).fit_resample(training_x, training_y)

                elif self.samples_balancing == "smote":

                    training_x, training_y = SMOTE(sampling_strategy=1.0).fit_resample(training_x, training_y)

                elif self.samples_balancing == "nearmiss":

                    training_x, training_y = NearMiss(sampling_strategy=1.0).fit_resample(training_x, training_y)

            self.base_classifiers[classifier_index] = self.base_classifiers[classifier_index].fit(
                training_x, training_y)

            training_score = self.base_classifiers[classifier_index].score(training_x, training_y)

            self.training_scores[classifier_index] = training_score

            target = "TRAINING SPECIALISTS LAYER: specialist " + str(specialist_classes[0]) + "-" + str(
                specialist_classes[1]) + ", training score, " + str(training_score)
            print(target)

        return self

    def predict(self, x):

        """
          This function computes the predictions for the given samples. Predictions are represented by a
          (n_samples, n_classes) matrix representing the sum of the probabilities for each class.

          | Pre-conditions: :py:meth:'OVOBaseLayer.fit'.
          | Post-conditions: predictions of the given data are obtained.
          | Main output: labels predicted for the given data.

        :param x: a containing the features of each sample for which prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples, n_classes).
        :raise: none.
        """

        predictions = np.zeros((x.shape[0], self.n_classes))

        for classifier_index, specialist_classes in enumerate(self.base_classifiers_dictionary):

            check_is_fitted(self.base_classifiers[classifier_index])

            predictions[:, specialist_classes[0]-1] += self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                       1] #* self.training_scores[classifier_index]
            predictions[:, specialist_classes[1]-1] += self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                       0] #* self.training_scores[classifier_index]

        return predictions

    def processing_history_predict(self, x, filename):

        """
          This function computes the predictions for the given samples and store information about the alterations
          experienced by the predictions themselves.

          | Pre-conditions: :py:meth:'OVOBaseLayer.fit'.
          | Post-conditions: predictions of the given data are obtained and info are stored in a file.
          | Main output: labels predicted for the given data.

        :param x: a matrix containing the features of each sample for which prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :param filename: name of the file in which info are stored.
        :type filename: str.
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples, n_classes).
        :raise: none.
        """

        predictions = np.zeros((x.shape[0], self.n_classes))
        not_weighted_predictions = np.zeros((x.shape[0], 2*self.n_base_classifiers))
        weighted_predictions = np.zeros((x.shape[0], 2*self.n_base_classifiers))

        for classifier_index, specialist_classes in enumerate(self.base_classifiers_dictionary):

            check_is_fitted(self.base_classifiers[classifier_index])

            predictions[:, specialist_classes[0] - 1] += self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         1] * self.training_scores[classifier_index]
            predictions[:, specialist_classes[1] - 1] += self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         0] * self.training_scores[classifier_index]
            not_weighted_predictions[:, 2*classifier_index] = self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         1]
            not_weighted_predictions[:, 2*classifier_index+1] = self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         0]
            weighted_predictions[:, 2*classifier_index] = self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         1] * self.training_scores[classifier_index]
            weighted_predictions[:, 2*classifier_index+1] = self.base_classifiers[classifier_index].predict_proba(x)[:,
                                                         0] * self.training_scores[classifier_index]

        columns_names_1 = np.array([str(c)+" - %d" % i for c in self.base_classifiers_dictionary for i in c])
        columns_names_2 = np.array([str(c)+" - %d x accuracy" % i for c in self.base_classifiers_dictionary for i in c])
        DatasetUtils.add_column_to_csv(filepath=filename, columns_values=not_weighted_predictions, columns_names=columns_names_1)
        DatasetUtils.add_column_to_csv(filepath=filename, columns_values=weighted_predictions, columns_names=columns_names_2)
        return predictions





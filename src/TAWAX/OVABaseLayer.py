import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from joblib import dump
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from src.Utils.ClassifiersManager import ClassifiersManager


class OVABaseLayer:

    """
      This class implements the base layer for the NAWAXOVA architecture. Base classifiers are trained in a
      1 vs ALL manner.
    """

    def __init__(self, n_classes, subject_id, models_path, results_path, samples_balancing):

        """
          This function initializes the OVABaseLayer (TAWAX version) object.

          | Pre-conditions: none.
          | Post-conditions: a new OVABaseLayer (TAWAX version) object is created.
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
            'random', 'clustering', 'smote', 'nearmiss', 'stratified_random'.
        :type samples_balancing: str.
        :return: none.
        :raise: none.
        """

        self.results_path = results_path
        self.models_path = models_path
        self.n_classes = n_classes
        self.subject_id = subject_id
        self.classes = np.arange(1, n_classes + 1)
        self.samples_balancing = samples_balancing

        # NON CLASS-DEPENDENT CLASSIFIERS
        classifiers_configuration_name = 'optuna_500_subject_%d_classes_%d' % (subject_id, n_classes)

        # UCI DATASETS
        # classifiers_configuration_name = 'mlp'

        self.base_classifiers = [ClassifiersManager.return_classifier(classifiers_configuration_name)
                                 for _ in np.arange(self.n_classes)]

        self.training_scores = np.zeros(self.n_classes)

    def fit(self, x, y):

        """
          This function fits the 1 vs ALL classifiers of the base layer on the given training set.

          | Pre-conditions: :py:meth:'OVABaseLayer.__init__'.
          | Post-conditions: the OVABaseLayer object is trained on the given data.
          | Main output: none.

        :param x: a matrix representing the features of each sample on which the layer
            is trained.
        :type x: ndarray (n_samples, n_features).
        :param ndarray y: an array of labels corresponding to the given features.
        :type y: ndarray (n_samples,).
        :return: self.
        :rtype: estimator.
        :raise: none.
        """

        x, y = check_X_y(x, y)

        for classifier_index, meta_class in enumerate(self.classes):

            training_x = np.array(x)
            training_y = np.where(y == meta_class, 1, 0)

            if self.samples_balancing == "random" or (self.samples_balancing == "stratified_random" and
               len(np.where(training_y == 0)[0]) < len(np.where(training_y == 1)[0])):

                training_x, training_y = RandomUnderSampler(sampling_strategy=1.0).fit_resample(training_x, training_y)

            elif self.samples_balancing == "clustering":

                training_x, training_y = ClusterCentroids(sampling_strategy=1.0).fit_resample(training_x, training_y)

            elif self.samples_balancing == "smote":

                training_x, training_y = SMOTE(sampling_strategy=1.0).fit_resample(training_x, training_y)

            elif self.samples_balancing == "nearmiss":

                training_x, training_y = NearMiss(sampling_strategy=1.0).fit_resample(training_x, training_y)

            elif self.samples_balancing == "stratified_random":

                num_samples_dict = {class_index: len(np.where(y == class_index)[0]) for class_index in self.classes}
                num_samples_minority_class = num_samples_dict[meta_class]
                num_samples_all_class = len(y) - num_samples_minority_class
                num_samples_vector = [
                    int((num_samples_dict[class_index] / num_samples_all_class) * num_samples_minority_class)
                    if meta_class != class_index else num_samples_minority_class for class_index in self.classes]
                training_x, training_y = RandomUnderSampler(
                    sampling_strategy=dict(zip(self.classes, num_samples_vector))).fit_resample(x, y)
                training_y = np.where(training_y == meta_class, 1, 0)

            training_x, training_y = check_X_y(training_x, training_y)

            self.base_classifiers[classifier_index] = self.base_classifiers[classifier_index].fit(training_x,
                                                                                                  training_y)

            training_score = self.base_classifiers[classifier_index].score(training_x, training_y)
            self.training_scores[classifier_index] = training_score

            target = "TRAINING META-CLASSIFIERS LAYER: meta-classifier " + str(meta_class) + ", training score, " + \
                     str(training_score)
            print(target)

    def predict(self, x):

        """
          This function computes the predictions for the given samples. Predictions are represented by a
          (n_samples, n_classes) matrix representing the probabilities for each class.

          | Pre-conditions: :py:meth:'OVABaseLayer.fit'.
          | Post-conditions: predictions of the given data are obtained.
          | Main output: labels predicted for the given data.

        :param x: a matrix containing the features of each sample for which
            prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples, n_classes).
        :raise: none.
        """

        predictions = np.empty((x.shape[0], self.n_classes))

        for classifier_index, meta_class in enumerate(self.classes):
            check_is_fitted(self.base_classifiers[classifier_index])

            predictions[:, classifier_index] = self.base_classifiers[classifier_index].predict_proba(x)[:, 1] * self.training_scores[classifier_index]

        return predictions



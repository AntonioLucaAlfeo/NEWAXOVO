import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from src.Utils.ClassifiersManager import ClassifiersManager
from src.Utils.DatasetUtils import DatasetUtils


class OVOBaseLayer:

    """
      This class implements the base layer for the DRCW-OVO architecture. Base classifiers are trained in
      a 1 vs 1 manner.
    """

    def __init__(self, n_classes, subject_id, models_path, results_path, samples_balancing):

        """
          This function initializes the OVOBaseLayer object.

          | Pre-conditions: none.
          | Post-conditions: a new OVOBaseLayer (DRCW version) object is created.
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
        self.classes = np.arange(1, n_classes + 1)

        self.n_base_classifiers = int(n_classes * (n_classes - 1) / 2)
        self.base_classifiers_dictionary = np.array([[i, j] for i in np.arange(start=1, stop=self.n_classes + 1)
                                                     for j in np.arange(start=i + 1, stop=self.n_classes + 1)])

        # NON-SPECIALIST DEPENDENT CLASSIFIERS
        classifiers_configurations_names = 'optuna_500_subject_%d_classes_%d' % (subject_id, n_classes)

        # UCI DATASETS
        #classifiers_configurations_names = 'mlp'

        self.base_classifiers = [ClassifiersManager.return_classifier(classifiers_configurations_names)
                                 for _ in np.arange(self.n_base_classifiers)]

        #SNIPPET FOR RETURNING CLASS-DEPENDENT CLASSIFIERS
        #self.base_classifiers = []
        #classifier_names = []

        #for specialist_classes in self.base_classifiers_dictionary:

        #    classifiers_configurations_names = 'random_1000_subject_%d_classifier_[%d_%d]' % (
        #        subject_id, specialist_classes[0], specialist_classes[1])
        #    classifier_names.append(classifiers_configurations_names)

        #self.base_classifiers = [ClassifiersManager.return_classifier(name) for name in classifier_names]

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

            # take only data related to current specialist classes
            relevant_rows = np.where(np.logical_or(y == specialist_classes[0], y == specialist_classes[1]))[0]  # [0] because we need only the indexes and not the additional info
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
          (n_samples, n_classes, n_classes) matrix representing the probabilities for each class. Considered a single
          sample, the (n_classes, n_classes) matrix represents the probabilities for each possible pair of classes.
          Sum of symmetric elements of this matrix is equal to 1.

          | Pre-conditions: :py:meth:'OVOBaseLayer.fit'.
          | Post-conditions: predictions of the given data are obtained.
          | Main output: labels predicted for the given data.

        :param x: a containing the features of each sample for which prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples, n_classes).
        :raise: none.
        """

        predictions = np.zeros((x.shape[0], self.n_classes, self.n_classes))

        for classifier_index, specialist_classes in enumerate(self.base_classifiers_dictionary):

            check_is_fitted(self.base_classifiers[classifier_index])

            predictions[:, specialist_classes[0]-1, specialist_classes[1]-1] = \
                self.base_classifiers[classifier_index].predict_proba(x)[:, 1]
            predictions[:, specialist_classes[1]-1, specialist_classes[0]-1] = \
                self.base_classifiers[classifier_index].predict_proba(x)[:, 0]

        return predictions

    def explain(self, x, class1, class2):

        """
          This function computes the explanations of the samples in x from the 1 vs 1 classifier class1[sample] vs
          class2[sample]. Explanations are obtained as SHAP values of the class with the lowest index between
          class1[sample] and class2[sample]: if class2[sample] < class1[sample], explanations are changed in sign.

          | Pre-conditions: :py:meth:'OVOBaseLayer.fit'.
          | Post-conditions: explanations of the given data are obtained from 1 vs 1 base classifiers.
          | Main output: explanations for the given data.

        :param x: a matrix containing the samples for which explanations are required.
        :type x: ndarray (n_samples*n_features).
        :param class1: a vector containing the primary class for each sample.
        :type class1: ndarray (n_samples,).
        :param class2: a vector containing the secondary class for each sample.
        :type class2: ndarray (n_samples,).
        :return: explanations of samples in x according to classifier class1[sample] vs class2[sample].
        :rtype: ndarray (n_samples*n_features).
        :raise: none.
        """

        specialist_classes = np.array(
            ['[%d %d]' % (c1, c2) if c1 < c2 else '[%d %d]' % (c2, c1) for c1, c2 in zip(class1, class2)])
        specialist_dictionary_string = np.array([str(c) for c in self.base_classifiers_dictionary])
        classifier_indexes = [np.argwhere(specialist_dictionary_string == specialist_classes[i])[0][0] for i in
                              np.arange(x.shape[0])]
        multipliers = np.array([1 if c1 < c2 else -1 for c1, c2 in zip(class1, class2)])
        dictionary = {}

        explanations = np.empty((x.shape[0], x.shape[1]))

        for sample_index, sample in enumerate(x):

            key = str(sample) + specialist_classes[sample_index]

            if key in dictionary:

                shap_values = dictionary[key]

            else:

                shap_values = self.explainers[classifier_indexes[sample_index]].shap_values(sample)[1]
                dictionary[key] = shap_values

            explanations[sample_index, :] = shap_values * multipliers[sample_index]

        return explanations

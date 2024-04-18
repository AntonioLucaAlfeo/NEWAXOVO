from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from src.Utils.ClassifiersManager import ClassifiersManager


class TAWAXSuperClassifierLayer:

    """
      This class implements the super classifier layer for the TAWAXOVA architecture. Super classifier is trained in a
      ALL vs ALL manner.
    """

    def __init__(self, n_classes, subject_id, models_path, results_path, classifier):

        """
          This function initializes the TAWAXSuperClassifierLayer object.

          | Pre-conditions: none.
          | Post-conditions: a new TAWAXSuperClassifierLayer object is created.
          | Main output: none.

        :param n_classes: number of classes of the given problem.
        :type n_classes: int (>0).
        :param subject_id: id of the subject considered.
        :type subject_id: int (between 1 and 9).
        :param models_path: path to directory in which models are saved.
        :type models_path: str.
        :param results_path: path to directory in which models are saved.
        :type results_path: str.
        :return: none.
        :raise: none.
        """

        self.results_path = results_path
        self.models_path = models_path
        self.n_classes = n_classes  # not used
        self.subject_id = subject_id

        self.super_classifier = MLPClassifier()

    def fit(self, x, y):

        """
          This function fits the super classifier on the given training set.

          | Pre-conditions: :py:meth:'TAWAXSuperClassifierLayer.__init__'.
          | Post-conditions: the TAWAXSuperClassifierLayer object is trained on the given data.
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

        self.super_classifier = self.super_classifier.fit(x, y)
        training_score = self.super_classifier.score(x, y)

        target = "TRAINING SUPER CLASSIFIER LAYER: super-classifier, training score, " + str(training_score)
        print(target)

        return self

    def predict(self, x):

        """
          This function computes the predictions for the given samples. Predictions are represented by a
          (n_samples,) array representing the final predictions.

          | Pre-conditions: :py:meth:'TAWAXSuperClassifierLayer.fit'.
          | Post-conditions: predictions of the given data are obtained.
          | Main output: labels predicted for the given data.

        :param x: a containing the features of each sample for which prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples,).
        :raise: none.
        """

        check_is_fitted(self.super_classifier)

        return self.super_classifier.predict(x)



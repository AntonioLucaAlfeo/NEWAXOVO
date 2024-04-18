import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class DataGenerator:

    """
      This class provides a way to create artificial datasets with user-defined characteristics.
    """

    def __init__(self, n_samples, n_features, n_informative, n_classes, relevance_decrease='linear'):

        """
          This function initializes the DataGenerator object.

          | Pre-conditions: none.
          | Post-conditions: a new DataGenerator object is created.
          | Main output: none.

        :param n_samples: number of samples.
        :type n_samples: int.
        :param n_features: total number of features.
        :type n_features: int.
        :param n_informative: number of informative features.
        :type n_informative: int.
        :param n_classes: number of classes (or labels) of the classification problem.
        :type n_classes: int.
        :param relevance_decrease: type of strategy used for decreasing feature relevance.
        :type relevance_decrease: str.
        :return: none.
        :raise: none.
        """

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_informative = n_informative
        self.relevance_decrease = relevance_decrease

    def make_data(self, random_state, n_clusters=2):

        """
          This function generates random data for a classification problem. It considers n_informative features and a
          certain number of redundant features. Noise is applied to redundant features in an increasing manner.

          | Pre-conditions: :py:meth:'DataGenerator.__init__'.
          | Post-conditions: a new artificial dataset is created.
          | Main output: the new dataset, split into features and labels.

        :param random_state: determines random number generation for dataset creation. Pass an int for reproducible
            output across multiple function calls.
        :type random_state: int, RandomState instance or None.
        :param n_clusters: number of clusters per class. Default 2.
        :type n_clusters: int (>0), optional.
        :return: features  and labels of the new problem.
        :rtype: ndarray (n_samples, n_features), ndarray (n_samples,).
        :raise:
        """

        # make_classification Generates a random n-class classification problem.
        try:
            x, y = make_classification(n_samples=self.n_samples, n_features=self.n_features,
                                       n_informative=self.n_informative,
                                       n_redundant=self.n_features - self.n_informative, n_repeated=0,
                                       n_classes=self.n_classes,
                                       n_clusters_per_class=n_clusters, flip_y=0.005, hypercube=False, shuffle=False,
                                       random_state=random_state)
        except ValueError:  # as err:

            return None, None

        # noise is increased in a linear way
        samples_deviation = np.std(np.ravel(x[:, self.n_informative:]))

        # noise_deviation = np.arange(start=0.1, stop=samples_deviation, step=samples_deviation / (self.n_features - self.n_informative))
        noise_deviation = np.linspace(start=0.1, stop=samples_deviation, num=(self.n_features - self.n_informative))

        x_final = x[:, 0:self.n_informative]  # take only columns related to informative features

        # add noise to non-informative features (a different gaussian noise is generated for each non informative features)
        for feature_index in np.arange(start=self.n_informative, stop=self.n_features):

            # Draw random samples from a normal (Gaussian) distribution. loc is the mean, scale the std deviation
            noise = np.random.normal(loc=0, scale=noise_deviation[feature_index - self.n_informative], size=self.n_samples)
            modified_feature = x[:, feature_index] + noise
            x_final = np.concatenate((x_final, np.expand_dims(modified_feature, axis=1)), axis=1)

        x_final = MinMaxScaler().fit_transform(x_final)
        y += 1

        return x_final, y

    def make_data_blob(self, n_equal, random_state):

        """
          This function computes centroid positions so that they will be orthogonal with respect to N random axis.

          | Pre-conditions: :py:meth:'DataGenerator.__init__'..
          | Post-conditions: a new artificial dataset is created.
          | Main output: the new dataset, split into features and labels.

        :param n_equal: number of features made up by the same value for all classes.
        :type n_equal: int (>0).
        :param random_state: determines random number generation for dataset creation. Pass an int for reproducible
            output across multiple function calls.
        :type random_state: int, RandomState instance or None.
        :return: features and labels of the new problem.
        :rtype: ndarray (n_samples, n_features), ndarray (n_samples,).
        :raise:
        """

        n_common = int((self.n_features - self.n_classes - n_equal) / self.n_classes)  # should we have n_common < n_classes

        centroid_positions = np.ones((self.n_classes, self.n_features)) * 2
        common_index = self.n_classes

        for class_index in range(self.n_classes):

            centroid_positions[class_index, class_index] = -2
            centroid_positions[class_index, common_index:common_index+n_common] = 0
            common_index += n_common

        x, y, centers = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=centroid_positions,
                                   cluster_std=1.0, shuffle=False, return_centers=True,
                                   random_state=random_state)  # Generate isotropic Gaussian blobs for clustering.

        y += 1
        return x, y, centroid_positions

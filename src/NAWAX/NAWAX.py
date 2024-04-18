import itertools
import time
import numpy as np
from sklearn.utils import check_X_y

from src.NAWAX.OVABaseLayer import OVABaseLayer
from src.Utils.DBScan import DBScan
from src.NAWAX.OVOBaseLayer import OVOBaseLayer
from src.Utils.DatasetUtils import DatasetUtils
from src.Utils.Plots import Plots
from src.Utils.ClassifiersManager import ClassifiersManager
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as ds
from sklearn.base import BaseEstimator


class NAWAX(BaseEstimator):

    """
      This class represents the skeleton for NAWAXOVO/NAWAXOVA architecture.
    """

    def __init__(self, n_classes, subject_id, models_path="models/", results_path="results/", n_neighbors=5,
                 base_layer_mode="1vs1", samples_balancing="random"):

        """
          This function initialises the NAWAX object using the given parameters. This will be the skeleton of
          NAWAXOVO/A architecture.

          | Pre-conditions: none.
          | Post-conditions: a new NAWAXOVO object is created.
          | Main output: none.

        :param n_classes: number of classes of the given problem.
        :type n_classes: int (>0).
        :param subject_id: id of the considered subject.
        :type subject_id: int (between 1 and 9).
        :param models_path: path in which models are saved. Default 'models/'.
        :type models_path: str, optional.
        :param results_path: path in which results are saved. Default 'results/'.
        :type results_path: str, optional.
        :param n_neighbors: number of neighbors used by NAWAXOVO/A's KNNs.
        :type n_neighbors: int (>0).
        :param base_layer_mode: one between '1vs1' and '1vsALL'. If '1vs1', base classifiers are trained in a 1 vs 1
            manner (NAWAXOVO). If '1vsALL', base classifiers are trained in a 1 vs ALL manner (NAWAXOVA).
            Default '1vs1'.
        :type base_layer_mode: str, optional.
        :param samples_balancing: balancing strategy adopted in case of non-balanced sub-problem. One between 'none',
            'random', 'stratified_random', 'clustering', 'smote', 'nearmiss'. In case of '1vs1', 'stratified_random' and
            'random' are the same (use 'random'). Default 'random'.
        :type samples_balancing: str, optional.
        :return: none.
        :raise: none.
        """

        if base_layer_mode != '1vs1' and base_layer_mode != '1vsALL':

            print("Base layer mode not supported ")
            return

        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        self.models_path = models_path
        self.results_path = results_path
        self.subject_id = subject_id
        self.n_neighbors = n_neighbors
        self.base_layer_mode = base_layer_mode
        self.samples_balancing = samples_balancing
        self.training_x = None
        self.training_y = None

        if base_layer_mode == "1vs1":

            self.base_path = "C:/Users/Luca/Desktop/NEWAXOVO/"

            self.base_classifiers_layer = OVOBaseLayer(models_path=models_path, results_path=results_path,
                                                       n_classes=n_classes, subject_id=subject_id,
                                                       samples_balancing=samples_balancing)
            self.n_base_classifiers = int(n_classes * (n_classes - 1) / 2)
            self.base_classifiers_dictionary = np.array([[i, j] for i in np.arange(start=1, stop=self.n_classes + 1)
                                                     for j in np.arange(start=i + 1, stop=self.n_classes + 1)])
        elif base_layer_mode == "1vsALL":

            self.base_path = "C:/Users/Luca/Desktop/NEWAXOVO/"

            self.base_classifiers_layer = OVABaseLayer(models_path=models_path, results_path=results_path,
                                                       n_classes=n_classes, subject_id=subject_id,
                                                       samples_balancing=samples_balancing)
            self.n_base_classifiers = n_classes

        knn_configuration_name = "knn_distance_cosine_weights_distance_k_%d" % n_neighbors
        self.knn_classifier = ClassifiersManager.return_classifier(knn_configuration_name)

    def fit(self, x, y):

        """
          This function fits the NAWAXOVO architecture on the given training set.

          | Pre-conditions: :py:meth:'NAWAX.__init__'.
          | Post-conditions: the NAWAXOVO object is trained on the given data.
          | Main output: none.

        :param x: a matrix representing the features of each sample on which the architecture
            is trained.
        :type x: ndarray (n_samples*n_features).
        :param y: an array of labels corresponding to the given features.
        :type y: ndarray (n_samples,).
        :return: self.
        :rtype: estimator.
        :raise: none.
        """

        x, y = check_X_y(x, y)

        self.training_x = x
        self.training_y = y

        self.base_classifiers_layer = self.base_classifiers_layer.fit(x, y)
        self.knn_classifier = self.knn_classifier.fit(x, y)

        return self

    def predict(self, x):

        """
          This function computes the predictions for the given samples.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: predictions of the given data are obtained.
          | Main output: labels predicted for the given data.

        :param x: a matrix containing the features of each sample for which
            prediction is needed.
        :type x: ndarray (n_samples*n_features).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples,).
        :raise: none.
        """

        base_layer_predictions = self.base_classifiers_layer.predict(x)
        knn_predictions = self.knn_classifier.predict_proba(x)

        if self.base_layer_mode == "1vs1":

            knn_weights = np.zeros((x.shape[0], 2*self.n_base_classifiers))

            for row_index in np.arange(knn_predictions.shape[0]):

                for specialist_index, specialist_classes in enumerate(self.base_classifiers_dictionary):

                    weight = np.mean([knn_predictions[row_index, specialist_classes[0]-1],
                                      knn_predictions[row_index, specialist_classes[1]-1]])

                    knn_weights[row_index, 2*specialist_index] = weight
                    knn_weights[row_index, 2 * specialist_index+1] = weight

            # multiply base predictions*knn weights
            weighted_predictions = np.multiply(base_layer_predictions, knn_weights)

            weighted_sum_predictions = np.zeros((x.shape[0], self.n_classes))

            for classifier_index, classes in enumerate(self.base_classifiers_dictionary):

                weighted_sum_predictions[:, classes[0]-1] += weighted_predictions[:, 2*classifier_index]
                weighted_sum_predictions[:, classes[1]-1] += weighted_predictions[:, 2*classifier_index+1]

            predictions = np.argmax(weighted_sum_predictions, axis=1) + 1

            return predictions

        elif self.base_layer_mode == "1vsALL":

            return np.argmax(np.multiply(base_layer_predictions, knn_predictions), axis=1)+1

    def processing_history_predict(self, x, y, ids):

        """
          This function computes the predictions for the given samples and store information about the alterations
          experienced by the predictions themselves.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: predictions of the given data are obtained and info are stored in a file.
          | Main output: labels predicted for the given data.

        :param x: a containing the features of each sample for which prediction is needed.
        :type x: ndarray (n_samples, n_features).
        :param y: an array of labels corresponding to the given features.
        :type y: ndarray (n_samples,).
        :param ids: an array containing the ids of the given samples.
        :type ids: ndarray (n_samples,).
        :return: the predictions for the given samples.
        :rtype: ndarray (n_samples, n_classes).
        :raise: none.
        """



        processing_history_filename = self.base_path + "history_predictions/nawax/processing_history_nawax_n_%s_resampling_%s_%s.csv" \
                                      % (self.n_neighbors, self.samples_balancing, time.strftime("%Y%m%d-%H%M%S"))
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_values=ids, columns_names=np.array(["ID"]))
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename,
                                       columns_values=np.repeat(self.subject_id, np.shape(x)[0]),
                                       columns_names=np.array(["SUBJECT"]))

        base_layer_predictions = self.base_classifiers_layer.processing_history_predict(x, processing_history_filename)
        knn_predictions = self.knn_classifier.predict_proba(x)

        knn_weights = np.zeros((x.shape[0], 2 * self.n_base_classifiers))

        for row_index in np.arange(knn_predictions.shape[0]):

            for specialist_index, specialist_classes in enumerate(self.base_classifiers_dictionary):
                knn_weights[row_index, 2 * specialist_index] = np.mean(
                    [knn_predictions[row_index, specialist_classes[0] - 1],
                     knn_predictions[row_index, specialist_classes[1] - 1]])
                knn_weights[row_index, 2 * specialist_index + 1] = np.mean(
                    [knn_predictions[row_index, specialist_classes[0] - 1],
                     knn_predictions[row_index, specialist_classes[1] - 1]])

        # multiply base predictions*knn weights
        weighted_predictions = np.multiply(base_layer_predictions, knn_weights)
        columns_names = np.array(
            [str(c) + " - %d x knn" % i for c in self.base_classifiers_dictionary for i in c])
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_names=columns_names, columns_values=weighted_predictions)

        weighted_sum_predictions = np.zeros((x.shape[0], self.n_classes))

        for classifier_index, classes in enumerate(self.base_classifiers_dictionary):
            weighted_sum_predictions[:, classes[0] - 1] += weighted_predictions[:, 2 * classifier_index]
            weighted_sum_predictions[:, classes[1] - 1] += weighted_predictions[:, 2 * classifier_index + 1]

        columns_names = np.array(["sumPw(1)", "sumPw(2)", "sumPw(3)", "sumPw(4)"])
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_names=columns_names,
                                       columns_values=weighted_sum_predictions)

        competence_scores_diff = np.zeros(x.shape[0])
        competence_scores_ratio = np.zeros(x.shape[0])

        for index in np.arange(x.shape[0]):
            c = weighted_sum_predictions[index, y[index]-1]
            all = np.sum(weighted_sum_predictions[index, :])
            others = all - c
            competence_scores_diff[index] = c - others
            competence_scores_ratio[index] = c / all

        predictions = np.argmax(weighted_sum_predictions, axis=1) + 1
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_values=predictions,
                                       columns_names=np.array(["PREDICTION"]))
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_values=y,
                                       columns_names=np.array(["LABEL"]))
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_values=competence_scores_diff,
                                       columns_names=np.array(["COMPETENCE REINFORCEMENT SCORE DIFF"]))
        DatasetUtils.add_column_to_csv(filepath=processing_history_filename, columns_values=competence_scores_ratio,
                                       columns_names=np.array(["COMPETENCE REINFORCEMENT SCORE RATIO"]))

        with open(self.base_path + "history_predictions/nawax/nawax_all_diff.csv", "a") as f:
            np.savetxt(f, competence_scores_diff, delimiter=",")

        with open(self.base_path + "history_predictions/nawax/nawax_all_ratio.csv", "a") as f:
            np.savetxt(f, competence_scores_ratio, delimiter=",")

        return predictions

    def explain(self, x, ids, y=None, contrastive_samples=None, contrastive_classes=None,
                mode="all_classes", q=3, k=5, p=0.04):

        """
          This function computes the explanations related to the samples x. For each sample, q nearest neighbors
          belonging to the majority class (excluded the predicted one)("majority_class"), to all classes (excluded
          the predicted one)("all_classes") or to the correct class ("error") are considered, and their explanations
          are computed. The most similar sample (in the explanations space) is taken as neighbor. Features are
          ordered by decreasing importance for the classification of the current sample, and the less important are
          discarded (according to the p parameter). For each feature, k points are taken from the interval
          [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)]. Cartesian product with
          the previously extracted points is computed, new artificial points are created, and their predictions are
          finally obtained. The algorithm stop when the prediction of an artificial point matches the class of the
          neighbor (or when the most important features have been analyzed).

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: explanations of the given data are obtained.
          | Main output: explanations for the given samples are saved in results_path/explanations_(time).

        :param x: a matrix of samples for which explanation is needed.
        :type x: ndarray (n_samples*n_features).
        :param ids: an array containing the ids of the samples in x.
        :type ids: ndarray (n_samples,).
        :param y: an array containing the true labels for the samples in x. Used only if mode == 'error'. Default None.
        :type y: ndarray (n_samples, ), optional.
        :param contrastive_samples: a matrix of samples representing the contrastive samples.
        :type contrastive_samples: ndarray (n_samples*n_features).
        :param contrastive_classes: an array containing the classes of the contrastive samples.
        :type contrastive_classes: ndarray (n_samples,).
        :param mode: one between 'all_classes', 'majority_class', 'error' and 'ids'. If 'all_classes', neighbors are
            chosen between elements of all classes except for the predicted one of the current sample. If
            'majority_class', neighbors are chosen between the elements of the majority class (except for the predicted
            one) of the current sample. If 'error', the function is equal to 'majority_class' with the difference that
            the correct label (specified in y) is considered as secondary class. Default 'all_classes'.
        :type mode: str, optional.
        :param q: number of neighbors considered for each sample. For each sample, q nearest neighbors belonging to the
            majority class (excluded the predicted one) are considered and their explanations are computed. Default
        :type q: int (>0), optional.
        :param k: number of points considered in the feature interval. At each iteration, k points are taken from
            the interval [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)]. Default
        :type k: int (>0), optional.
        :param p: percentage of features considered (a number in [0, 1]). The algorithm stops after examining the
            p*n_features most important features. Default 0.04.
        :type p: float (in [0, 1]), optional.
        :return: none.
        :raise: none.
        """

        if mode == "all_classes":

            return self.__explain_all_classes(x=x, ids=ids, q=q, k=k, p=p)

        elif mode == "majority_class":

            return self.__explain_majority_class(x=x, ids=ids, q=q, k=k, p=p)

        elif mode == "error":

            return self.__explain_error(x=x, ids=ids, y=y, q=q, k=k, p=p)

        elif mode == "ids":

            return self.__explain_ids(x=x, ids=ids, contrastive_samples=contrastive_samples,
                                     contrastive_classes=contrastive_classes, k=k, p=p)

    def __explain_majority_class(self, x, ids, q, k, p):

        """
          This function computes the explanations related to the samples x. For each sample, q nearest neighbors
          belonging to the majority class (excluded the predicted one) are considered, and their explanations are
          computed. The most similar sample (in the explanations space) is taken as neighbor. Features are ordered by
          decreasing importance for the classification of the current sample, and the less important are discarded
          (according to the p parameter). For each feature, k points are taken from the interval [min(sample_feature,
          neighbor_feature), max(sample_feature, neighbor_feature)]. Cartesian product with the previously extracted
          points is computed, new artificial points are created, and their predictions are finally obtained.
          The algorithm stop when the prediction of an artificial point matches the class of the neighbor (or when
          the most important features have been analyzed). Private method.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: explanations of the given data are obtained using the 'majority class' approach.
          | Main output: explanations for the given samples are saved in results_path/explanations_(time).

        :param x: a matrix of samples for which explanation is needed.
        :type x: ndarray (n_samples*n_features).
        :param ids: an array containing the ids of the samples in x.
        :type ids: ndarray (n_samples,).
        :param q: number of neighbors considered for each sample. For each sample, q nearest neighbors belonging to the
            majority class (excluded the predicted one) are considered and their explanations are computed.
        :type q: int (>0).
        :param k: number of points considered in the feature interval. At each iteration, k points are taken from
            the interval [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)].
        :type k: int (>0).
        :param p: percentage of features considered (a number in [0, 1]). The algorithm stops after examining the
            p*n_features most important features.
        :type p: float (in [0, 1]).
        :return: none.
        :raise: none.
        """

        if self.base_layer_mode == "1vs1":

            base_layer_predictions = self.base_classifiers_layer.predict(x)
            knn_predictions = self.knn_classifier.predict_proba(x)

            knn_weights = np.zeros((x.shape[0], 2 * self.n_base_classifiers))

            for row_index in np.arange(knn_predictions.shape[0]):

                for specialist_index, specialist_classes in enumerate(self.base_classifiers_dictionary):
                    weight = np.mean([knn_predictions[row_index, specialist_classes[0] - 1],
                                          knn_predictions[row_index, specialist_classes[1] - 1]])

                    knn_weights[row_index, 2 * specialist_index] = weight
                    knn_weights[row_index, 2 * specialist_index + 1] = weight

            # multiply base predictions*knn weights
            weighted_predictions = np.multiply(base_layer_predictions, knn_weights)

            weighted_sum_predictions = np.zeros((x.shape[0], self.n_classes))

            for classifier_index, classes in enumerate(self.base_classifiers_dictionary):
                weighted_sum_predictions[:, classes[0] - 1] += weighted_predictions[:, 2 * classifier_index]
                weighted_sum_predictions[:, classes[1] - 1] += weighted_predictions[:, 2 * classifier_index + 1]

            # take predicted class and second predicted class for each sample
            predicted_classes = np.argsort(-weighted_sum_predictions, axis=1)[:, :2] + 1

            # save training set index reordering according to KNN for each sample
            neighbor_indexes = self.knn_classifier.kneighbors(x, self.knn_classifier.n_samples_fit_,
                                                              return_distance=False)

            # explanations of the samples considering only the appropriate specialist classifier (C1 vs C2)
            specialist_explanations = self.base_classifiers_layer.explain(x, predicted_classes[:, 0],
                                                                          predicted_classes[:, 1])

            # number of features considered
            n_considered_features = int(x.shape[1] * p)

            # name of the file in which explanations are stored
            output_filename = "explanations_nawaxovo_mode_majority_subject_%d_%s.txt" % (self.subject_id,
                                                                                time.strftime("%Y%m%d-%H%M%S"))

            for sample_index, sample in enumerate(x):

                print("---------- EXPLAINING SAMPLE %d ----------" % (ids[sample_index]))
                start_time = time.time()

                # flag for stopping and passing to the next neighbor
                flag = False

                # reorder training set elements according to KNN results
                neighbors_reordered = self.training_x[neighbor_indexes[sample_index, :], :]
                labels_reordered = self.training_y[neighbor_indexes[sample_index, :]]

                # filter out elements whose class is different from C2 and take the first q elements
                relevant_rows = np.where(labels_reordered == predicted_classes[sample_index, 1])[0]
                best_neighbors = np.array(neighbors_reordered[relevant_rows, :][:q, :])

                # get neighbors explanations from the appropriate specialist classifier
                neighbors_explanations = self.base_classifiers_layer.explain(best_neighbors,
                                                                             np.repeat(
                                                                                 predicted_classes[sample_index, 0], q),
                                                                             np.repeat(
                                                                                 predicted_classes[sample_index, 1], q))

                # get cosine similarities between sample explanation and neighbors explanations
                cosine_similarities = cosine_similarity(
                    np.expand_dims(specialist_explanations[sample_index, :], axis=0),
                    neighbors_explanations)

                # get the closest neighbor
                neighbor = best_neighbors[np.argmax(cosine_similarities[0])]

                # obtain the abs of shap values of current sample in order to find the feature importance
                vals_abs = np.abs(specialist_explanations[sample_index, :])

                # this vector contains the indexes of the features, from the most important to less important
                features_reordering = np.argsort(-vals_abs)[:n_considered_features]

                # obtain the names of the most important features (used in plot)
                most_important_features = np.array([DatasetUtils.return_feature_name(i)
                                                    for i in np.argsort(-vals_abs)[:n_considered_features * 2]])

                # obtain the highest abs of shap values
                most_important_features_abs = np.sort(vals_abs)[::-1][:n_considered_features * 2]

                # this matrix will contain the features created at each step, for each of the n_intervals points
                # considered. Each row will contains new values for a single feature.
                features_new_values_matrix = np.empty((0, k))

                for i, feature_index in enumerate(features_reordering):

                    print("ANALYZING FEATURE %d, CORRESPONDING TO INDEX %d (%s) (ABS SHAP %.4f)" % (
                        (i + 1), feature_index, most_important_features[i], most_important_features_abs[i]))

                    # get feature value for sample and neighbor
                    feature_value_sample = sample[feature_index]
                    feature_value_neighbor = neighbor[feature_index]

                    # feature1 will represent the lower bound of the interval, feature2 the upper bound
                    feature1 = min(feature_value_sample, feature_value_neighbor)
                    feature2 = max(feature_value_sample, feature_value_neighbor)

                    # generate k equidistant point in the interval (feature1, feature2)
                    points = np.linspace(start=feature1, stop=feature2, num=k + 1, endpoint=False)[1:].reshape(1, -1)

                    # if analyzing second most important feature (or third ecc.), need to create the cartesian
                    # product with the values of the previously analyzed features
                    if i >= 1:

                        auxiliary_matrix = np.concatenate((features_new_values_matrix, points), axis=0)
                        features_new_values = np.array(list(itertools.product(*auxiliary_matrix)))

                    # first iteration: new values correspond to the new generated points
                    else:

                        features_new_values = np.array(points).reshape(-1, 1)

                    # add new features to the features_new_values_matrix object
                    features_new_values_matrix = np.append(features_new_values_matrix, points, axis=0)

                    # take original sample features (until the i-th most important)
                    sample_features = np.array([sample[features_reordering[j]] for j in np.arange(i + 1)]).reshape(1,
                                                                                                                   -1)

                    # reorder new features values accordingly to the distance to the original feature values
                    features_new_values = features_new_values[np.argsort(ds.cdist(sample_features, features_new_values,
                                                                                  metric="euclidean")[0]), :]

                    # each values is a list containing the new feature values
                    for u, values in enumerate(features_new_values):

                        # copy the original sample
                        new_sample = np.array(sample).reshape(1, -1)

                        # for each feature until the i-th most important, update the value in new_sample
                        for j, value in enumerate(values):
                            new_sample[0, features_reordering[j]] = value

                        prediction = self.predict(new_sample)

                        # end: pass to the next sample
                        if prediction == predicted_classes[sample_index, 1]:

                            print("Prediction changed to C2 for feature values %s: end" % (
                                dict(zip(features_reordering[:i + 1], values))))
                            flag = True
                            features_names = np.array([])
                            end_time = time.time()

                            explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d.\n\tPrediction will turn to %d by varying " \
                                          "the following features of the following amounts:\n" % (
                                          ids[sample_index], end_time - start_time,
                                          predicted_classes[sample_index, 0], predicted_classes[sample_index, 1])

                            for j, value in enumerate(values):
                                explanation += "\t\tPDS band %s channel %s: %+.4f%%\n" % (
                                    DatasetUtils.return_band(features_reordering[j]),
                                    DatasetUtils.return_electrode_number(features_reordering[j]),
                                    ((value - sample[features_reordering[j]]) * 100 / max(
                                        sample[features_reordering[j]], 1e-8)))
                                features_names = np.append(features_names,
                                                           DatasetUtils.return_feature_name(features_reordering[j]))

                            db = DBScan(min_pts=2, dataset=features_names).dbscan()
                            if db.get_n_clusters() >= 1:
                                explanation += "\tClusters.\n" + db.get_clusters()
                            break

                        elif prediction == predicted_classes[sample_index, 0]:

                            print("Prediction is always the same for feature values %s: research continues" % (
                                dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction not changed using given parameters.\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index, 0])
                                flag = True

                        else:

                            print(
                                "Prediction is changed but not into the desired class for feature values %s: research continues" % (
                                    dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction changed but not into the desired class using given parameters.\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index, 0])
                                flag = True

                    if flag:
                        with open(self.base_path + "explanations/" + output_filename, "a", encoding="utf-8") as file_object:
                            file_object.write(explanation)
                            file_object.close()

                        break

        elif self.base_layer_mode == "1vsALL":

            print("No implementation yet")

    def __explain_all_classes(self, x, ids, q, k, p):

        """
          This function computes the explanations related to the samples x. For each sample, q nearest neighbors
          belonging to all classes (excluded the predicted one) are considered, and their explanations are computed.
          The most similar sample (in the explanations space) is taken as neighbor. Features are ordered by
          decreasing importance for the classification of the current sample, and the less important are discarded
          (according to the p parameter). For each feature, k points are taken from the interval [min(sample_feature,
          neighbor_feature), max(sample_feature, neighbor_feature)]. Cartesian product with the previously extracted
          points is computed, new artificial points are created, and their predictions are finally obtained.
          The algorithm stop when the prediction of an artificial point matches the class of the neighbor
          (or when the most important features have been analyzed). Private method.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: explanations of the given data are obtained using the 'all classes' approach.
          | Main output: explanations for the given samples are saved in results_path/explanations_(time).

        :param x: a matrix of samples for which explanation is needed.
        :type x: ndarray (n_samples*n_features).
        :param ids: an array containing the ids of the samples in x.
        :type ids: ndarray (n_samples,).
        :param q: number of neighbors considered for each sample. For each sample, q nearest neighbors belonging to the
            majority class (excluded the predicted one) are considered and their explanations are computed.
        :type q: int (>0).
        :param k: number of points considered in the feature interval. At each iteration, k points are taken from
            the interval [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)].
        :type k: int (>0).
        :param p: percentage of features considered (a number in [0, 1]). The algorithm stops after examining the
            p*n_features most important features.
        :type p: float (in [0, 1]).
        :return: none.
        :raise: none.
        """

        if self.base_layer_mode == "1vs1":

            # qui non serve la seconda classe predetta
            # take predicted class and second predicted class for each sample
            predicted_classes = self.predict(x)

            # save training set index reordering according to KNN for each sample
            neighbor_indexes = self.knn_classifier.kneighbors(x, self.knn_classifier.n_samples_fit_,
                                                              return_distance=False)

            # number of features considered
            n_considered_features = int(x.shape[1] * p)

            # name of the file in which explanations are stored
            output_filename = "explanations_nawaxovo_mode_allclasses_subject_%d_%s.txt" % (self.subject_id,
                                                                                  time.strftime("%Y%m%d-%H%M%S"))

            for sample_index, sample in enumerate(x):

                print("---------- EXPLAINING SAMPLE %d ----------" % (ids[sample_index]))
                start_time = time.time()

                # flag for stopping and passing to the next neighbor
                flag = False

                # reorder training set elements according to KNN results
                neighbors_reordered = self.training_x[neighbor_indexes[sample_index, :], :]
                labels_reordered = self.training_y[neighbor_indexes[sample_index, :]]

                # filter out elements whose class is C1 (take elements of all classes except C1)
                relevant_rows = np.where(labels_reordered != predicted_classes[sample_index])[0]
                best_neighbors = np.array(neighbors_reordered[relevant_rows, :][:q, :])
                best_neighbors_labels = np.array(labels_reordered[relevant_rows][:q])

                # get neighbors explanations from the appropriate specialist classifier
                # C1 is the predicted class for the current sample
                # C2 is represented by the classes of the neighbors
                neighbors_explanations = self.base_classifiers_layer.explain(best_neighbors,
                                                                             np.repeat(predicted_classes[sample_index],
                                                                                       q),
                                                                             best_neighbors_labels)

                # explanation of the sample considering only the appropriate specialist classifier (C1 vs C2)
                sample_explanation = self.base_classifiers_layer.explain(np.tile(sample, (q, 1)),
                                                                         np.repeat(predicted_classes[sample_index], q),
                                                                         best_neighbors_labels)
                # get cosine similarities between sample explanation and neighbors explanations
                cosine_similarities = cosine_similarity(sample_explanation, neighbors_explanations).diagonal()

                # get the closest neighbor and its label
                neighbor = best_neighbors[np.argmax(cosine_similarities), :]
                neighbor_label = best_neighbors_labels[np.argmax(cosine_similarities)]

                # obtain the abs of shap values of current sample in order to find the feature importance
                vals_abs = np.abs(sample_explanation[np.argmax(cosine_similarities), :])

                # this vector contains the indexes of the features, from the most important to less important
                features_reordering = np.argsort(-vals_abs)[:n_considered_features]

                # obtain the names of the most important features (used in plot)
                most_important_features = np.array([DatasetUtils.return_feature_name(i)
                                                    for i in np.argsort(-vals_abs)[:n_considered_features * 2]])

                # obtain the highest abs of shap values
                most_important_features_abs = np.sort(vals_abs)[::-1][:n_considered_features * 2]

                # this matrix will contain the features created at each step, for each of the n_intervals points
                # considered. Each row will contains new values for a single feature.
                features_new_values_matrix = np.empty((0, k))

                for i, feature_index in enumerate(features_reordering):

                    print("ANALYZING FEATURE %d, CORRESPONDING TO INDEX %d (%s) (ABS SHAP %.4f)" % (
                        (i + 1), feature_index, most_important_features[i], most_important_features_abs[i]))

                    # get feature value for sample and neighbor
                    feature_value_sample = sample[feature_index]
                    feature_value_neighbor = neighbor[feature_index]

                    # feature1 will represent the lower bound of the interval, feature2 the upper bound
                    feature1 = min(feature_value_sample, feature_value_neighbor)
                    feature2 = max(feature_value_sample, feature_value_neighbor)

                    # generate k equidistant point in the interval (feature1, feature2)
                    points = np.linspace(start=feature1, stop=feature2, num=k + 1, endpoint=False)[1:].reshape(1, -1)

                    # if analyzing second most important feature (or third ecc.), need to create the cartesian
                    # product with the values of the previously analyzed features
                    if i >= 1:

                        auxiliary_matrix = np.concatenate((features_new_values_matrix, points), axis=0)
                        features_new_values = np.array(list(itertools.product(*auxiliary_matrix)))

                    # first iteration: new values correspond to the new generated points
                    else:

                        features_new_values = np.array(points).reshape(-1, 1)

                    # add new features to the features_new_values_matrix object
                    features_new_values_matrix = np.append(features_new_values_matrix, points, axis=0)

                    # take original sample features (until the i-th most important)
                    sample_features = np.array([sample[features_reordering[j]] for j in np.arange(i + 1)]).reshape(1,
                                                                                                                   -1)

                    # reorder new features values accordingly to the distance to the original feature values
                    features_new_values = features_new_values[np.argsort(ds.cdist(sample_features, features_new_values,
                                                                                  metric="euclidean")[0]), :]

                    # each values is a list containing the new feature values
                    for u, values in enumerate(features_new_values):

                        # copy the original sample
                        new_sample = np.array(sample).reshape(1, -1)

                        # for each feature until the i-th most important, update the value in new_sample
                        for j, value in enumerate(values):
                            new_sample[0, features_reordering[j]] = value

                        prediction = self.predict(new_sample)

                        # end: pass to the next sample
                        if prediction == neighbor_label:

                            print("Prediction changed to C2 for feature values %s: end" % (
                                dict(zip(features_reordering[:i + 1], values))))
                            flag = True
                            features_names = np.array([])
                            end_time = time.time()

                            explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d.\n\tPrediction will turn to %d by varying " \
                                          "the following features of the following amounts:\n" % \
                                          (ids[sample_index], end_time - start_time, predicted_classes[sample_index],
                                           neighbor_label)

                            for j, value in enumerate(values):
                                explanation += "\t\tPDS band %s channel %s: %+.4f%%\n" % (
                                    DatasetUtils.return_band(features_reordering[j]),
                                    DatasetUtils.return_electrode_number(features_reordering[j]),
                                    ((value - sample[features_reordering[j]]) * 100 / max(
                                        sample[features_reordering[j]], 1e-8)))
                                features_names = np.append(features_names,
                                                           DatasetUtils.return_feature_name(features_reordering[j]))

                            db = DBScan(min_pts=2, dataset=features_names).dbscan()

                            if db.get_n_clusters() >= 1:
                                explanation += "\tClusters.\n" + db.get_clusters()
                            break

                        elif prediction == predicted_classes[sample_index]:

                            print("Prediction is always the same for feature values %s: research continues" % (
                                dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction not changed using given parameters\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True

                        else:

                            print(
                                "Prediction is changed but not into the desired class for feature values %s: research continues" % (
                                    dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction changed but not into the desired class using given parameters\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True
                    if flag:
                        with open(self.base_path + "explanations/" + output_filename, "a", encoding="utf-8") as file_object:
                            file_object.write(explanation)
                            file_object.close()

                        break

        elif self.base_layer_mode == "1vsALL":

            print("No implementation yet")

    def __explain_error(self, x, ids, y, q, k, p):

        """
          This function computes the explanations related to the samples x. For each sample, q nearest neighbors
          belonging to the correct class (y) are considered, and their explanations are computed. The most
          similar sample (in the explanations space) is taken as neighbor. Features are ordered by decreasing
          importance for the classification of the current sample, and the less important are discarded (according to
          the p parameter). For each feature, k points are taken from the interval [min(sample_feature,
          neighbor_feature), max(sample_feature, neighbor_feature)]. Cartesian product with the previously
          extracted points is computed, new artificial points are created, and their predictions are finally obtained.
          The algorithm stop when the prediction of an artificial point matches the class of the neighbor
          (or when the most important features have been analyzed). Private method.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: explanations of the given data are obtained using the 'error' approach.
          | Main output: explanations for the given samples are saved in results_path/explanations_(time).

        :param x: a matrix of samples for which explanation is needed.
        :type x: ndarray (n_samples*n_features).
        :param ids: an array containing the ids of the samples in x.
        :type ids: ndarray (n_samples,).
        :param y: an array containing the true labels for the samples in x.
        :type y: ndarray (n_samples, ).
        :param q: number of neighbors considered for each sample. For each sample, q nearest neighbors belonging to the
            majority class (excluded the predicted one) are considered and their explanations are computed.
        :type q: int (>0).
        :param k: number of points considered in the feature interval. At each iteration, k points are taken from
            the interval [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)].
        :type k: int (>0).
        :param p: percentage of features considered (a number in [0, 1]). The algorithm stops after examining the
            p*n_features most important features.
        :type p: float (in [0, 1]).
        :return: none.

        :raise: none.
        """

        if self.base_layer_mode == "1vs1":

            # take predicted class for each sample
            predicted_classes = self.predict(x)

            # save training set index reordering according to KNN for each sample
            neighbor_indexes = self.knn_classifier.kneighbors(x, self.knn_classifier.n_samples_fit_,
                                                              return_distance=False)

            # explanations of the given samples from the appropriate base classifier
            samples_explanations = self.base_classifiers_layer.explain(x, predicted_classes, y)

            # number of features considered
            n_considered_features = int(x.shape[1] * p)

            # name of the file in which explanations are stored
            output_filename = "explanations_nawaxovo_mode_error_subject_%d_%s.txt" % (self.subject_id,
                                                                             time.strftime("%Y%m%d-%H%M%S"))

            for sample_index, sample in enumerate(x):

                print("---------- EXPLAINING SAMPLE %d ----------" % (ids[sample_index]))

                start_time = time.time()

                # flag for stopping and passing to the next neighbor
                flag = False

                # reorder training set elements according to KNN results
                neighbors_reordered = self.training_x[neighbor_indexes[sample_index, :], :]
                labels_reordered = self.training_y[neighbor_indexes[sample_index, :]]

                # filter out elements whose class is different from the real class of the current sample
                relevant_rows = np.where(labels_reordered == y[sample_index])[0]
                best_neighbors = np.array(neighbors_reordered[relevant_rows, :][:q, :])

                # get neighbors explanations from the appropriate specialist classifier
                # C1 is the predicted class for the current sample
                # C2 is represented by the real class of the current sample
                neighbors_explanations = self.base_classifiers_layer.explain(best_neighbors,
                                                                             np.repeat(predicted_classes[sample_index],
                                                                                       q),
                                                                             np.repeat(y[sample_index], q))

                # get cosine similarities between sample explanation and neighbors explanations
                cosine_similarities = cosine_similarity(
                    np.expand_dims(samples_explanations[sample_index, :], axis=0),
                    neighbors_explanations)

                # get the closest neighbor
                neighbor = best_neighbors[np.argmax(cosine_similarities[0])]

                # obtain the abs of shap values of current sample in order to find the feature importance
                vals_abs = np.abs(samples_explanations[sample_index, :])

                # this vector contains the indexes of the features, from the most important to less important
                features_reordering = np.argsort(-vals_abs)[:n_considered_features]

                # obtain the names of the most important features (used in plot)
                most_important_features = np.array([DatasetUtils.return_feature_name(i)
                                                    for i in np.argsort(-vals_abs)[:n_considered_features * 2]])

                # obtain the highest abs of shap values
                most_important_features_abs = np.sort(vals_abs)[::-1][:n_considered_features * 2]

                # this matrix will contain the features created at each step, for each of the n_intervals points
                # considered. Each row will contains new values for a single feature.
                features_new_values_matrix = np.empty((0, k))

                for i, feature_index in enumerate(features_reordering):

                    print("ANALYZING FEATURE %d, CORRESPONDING TO INDEX %d (%s) (ABS SHAP %.4f)" % ((i + 1),
                                                                                                    feature_index,
                                                                                                    most_important_features[
                                                                                                        i],
                                                                                                    most_important_features_abs[
                                                                                                        i]))

                    # get feature value for sample and neighbor
                    feature_value_sample = sample[feature_index]
                    feature_value_neighbor = neighbor[feature_index]

                    # feature1 will represent the lower bound of the interval, feature2 the upper bound
                    feature1 = min(feature_value_sample, feature_value_neighbor)
                    feature2 = max(feature_value_sample, feature_value_neighbor)

                    # generate k equidistant point in the interval (feature1, feature2)
                    points = np.linspace(start=feature1, stop=feature2, num=k + 1, endpoint=False)[1:].reshape(1, -1)

                    # if analyzing second most important feature (or third ecc.), need to create the cartesian
                    # product with the values of the previously analyzed features
                    if i >= 1:

                        auxiliary_matrix = np.concatenate((features_new_values_matrix, points), axis=0)
                        features_new_values = np.array(list(itertools.product(*auxiliary_matrix)))

                    # first iteration: new values correspond to the new generated points
                    else:

                        features_new_values = np.array(points).reshape(-1, 1)

                    # add new features to the features_new_values_matrix object
                    features_new_values_matrix = np.append(features_new_values_matrix, points, axis=0)

                    # take original sample features (until the i-th most important)
                    sample_features = np.array([sample[features_reordering[j]] for j in np.arange(i + 1)]).reshape(1,
                                                                                                                   -1)

                    # reorder new features values accordingly to the distance to the original feature values
                    features_new_values = features_new_values[np.argsort(ds.cdist(sample_features, features_new_values,
                                                                                  metric="euclidean")[0]), :]

                    # each values is a list containing the new feature values
                    for u, values in enumerate(features_new_values):

                        # copy the original sample
                        new_sample = np.array(sample).reshape(1, -1)

                        # for each feature until the i-th most important, update the value in new_sample
                        for j, value in enumerate(values):
                            new_sample[0, features_reordering[j]] = value

                        prediction = self.predict(new_sample)

                        # end: pass to the next sample
                        if prediction == y[sample_index]:

                            print("Prediction changed to C2 for feature values %s: end" % (
                                dict(zip(features_reordering[:i + 1], values))))
                            flag = True
                            features_names = np.array([])
                            end_time = time.time()

                            explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d.\n\tPrediction will turn to the real class %d by varying " \
                                          "the following features of the following amounts:\n" % \
                                          (ids[sample_index], end_time - start_time, predicted_classes[sample_index],
                                           y[sample_index])

                            for j, value in enumerate(values):
                                explanation += "\t\tPDS band %s channel %s: %+.4f%%\n" % (
                                    DatasetUtils.return_band(features_reordering[j]),
                                    DatasetUtils.return_electrode_number(features_reordering[j]),
                                    ((value - sample[features_reordering[j]]) * 100 / max(
                                        sample[features_reordering[j]], 1e-8)))
                                features_names = np.append(features_names,
                                                           DatasetUtils.return_feature_name(features_reordering[j]))

                            db = DBScan(min_pts=2, dataset=features_names).dbscan()
                            if db.get_n_clusters() >= 1:
                                explanation += "\tClusters.\n" + db.get_clusters()
                            break

                        elif prediction == predicted_classes[sample_index]:

                            print("Prediction is always the same for feature values %s: research continues" % (
                                dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction not changed using given parameters\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True
                                break

                        else:

                            print(
                                "Prediction is changed but not into the desired class for feature values %s: research continues" % (
                                    dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction changed but not into the desired class using given parameters\n" % (
                                ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True
                                break
                    if flag:
                        with open(self.base_path + "explanations/" + output_filename, "a", encoding="utf-8") as file_object:
                            file_object.write(explanation)
                            file_object.close()

                        break

        elif self.base_layer_mode == "1vsALL":

            print("No implementation yet")

    def __explain_ids(self, x, ids, contrastive_samples, contrastive_classes, k, p):

        """
          This function computes the explanations related to the samples x. For each sample, q nearest neighbors
          belonging to the correct class (y) are considered, and their explanations are computed. The most
          similar sample (in the explanations space) is taken as neighbor. Features are ordered by decreasing
          importance for the classification of the current sample, and the less important are discarded (according to
          the p parameter). For each feature, k points are taken from the interval [min(sample_feature,
          neighbor_feature), max(sample_feature, neighbor_feature)]. Cartesian product with the previously
          extracted points is computed, new artificial points are created, and their predictions are finally obtained.
          The algorithm stop when the prediction of an artificial point matches the class of the neighbor
          (or when the most important features have been analyzed). Private method.

          | Pre-conditions: :py:meth:'NAWAX.fit'.
          | Post-conditions: explanations of the given data are obtained using the 'error' approach.
          | Main output: explanations for the given samples are saved in results_path/explanations_(time).

           :param x: a matrix of samples for which explanation is needed.
           :type x: ndarray (n_samples*n_features).
           :param ids: an array containing the ids of the samples in x.
           :type ids: ndarray (n_samples,).
           :param contrastive_samples: a matrix of samples representing the contrastive samples.
           :type contrastive_samples: ndarray (n_samples*n_features).
           :param contrastive_classes: an array containing the classes of the contrastive samples.
           :type contrastive_classes: ndarray (n_samples,).
           :param k: number of points considered in the feature interval. At each iteration, k points are taken from
               the interval [min(sample_feature, neighbor_feature), max(sample_feature, neighbor_feature)].
           :type k: int (>0).
           :param p: percentage of features considered (a number in [0, 1]). The algorithm stops after examining the
               p*n_features most important features.
           :type p: float (in [0, 1]).
           :return: none.

           :raise: none.
           """

        if self.base_layer_mode == "1vs1":

            # take predicted class and second predicted class for each sample
            predicted_classes = self.predict(x)

            # explanations of the given samples from the appropriate base classifier
            samples_explanations = self.base_classifiers_layer.explain(x, predicted_classes, contrastive_classes)

            # number of features considered
            n_considered_features = int(x.shape[1] * p)

            # name of the file in which explanations are stored
            output_filename = "explanations_nawaxovo_mode_ids_subject_%d_%s.txt" % (self.subject_id,
                                                                           time.strftime("%Y%m%d-%H%M%S"))

            for sample_index, sample in enumerate(x):

                print("---------- EXPLAINING SAMPLE %d ----------" % (ids[sample_index]))

                start_time = time.time()

                # flag for stopping and passing to the next neighbor
                flag = False

                # get the closest neighbor
                neighbor = contrastive_samples[sample_index]

                # obtain the abs of shap values of current sample in order to find the feature importance
                vals_abs = np.abs(samples_explanations[sample_index, :])

                # this vector contains the indexes of the features, from the most important to less important
                features_reordering = np.argsort(-vals_abs)[:n_considered_features]

                # obtain the names of the most important features (used in plot)
                most_important_features = np.array([DatasetUtils.return_feature_name(i)
                                                    for i in np.argsort(-vals_abs)[:n_considered_features * 2]])

                # obtain the highest abs of shap values
                most_important_features_abs = np.sort(vals_abs)[::-1][:n_considered_features * 2]

                # this matrix will contain the features created at each step, for each of the n_intervals points
                # considered. Each row will contains new values for a single feature.
                features_new_values_matrix = np.empty((0, k))

                for i, feature_index in enumerate(features_reordering):

                    print("ANALYZING FEATURE %d, CORRESPONDING TO INDEX %d (%s) (ABS SHAP %.4f)" % ((i + 1),
                                                                                                    feature_index,
                                                                                                    most_important_features[
                                                                                                        i],
                                                                                                    most_important_features_abs[
                                                                                                        i]))

                    # get feature value for sample and neighbor
                    feature_value_sample = sample[feature_index]
                    feature_value_neighbor = neighbor[feature_index]

                    # feature1 will represent the lower bound of the interval, feature2 the upper bound
                    feature1 = min(feature_value_sample, feature_value_neighbor)
                    feature2 = max(feature_value_sample, feature_value_neighbor)

                    # generate k equidistant point in the interval (feature1, feature2)
                    points = np.linspace(start=feature1, stop=feature2, num=k + 1, endpoint=False)[1:].reshape(1, -1)

                    # if analyzing second most important feature (or third ecc.), need to create the cartesian
                    # product with the values of the previously analyzed features
                    if i >= 1:

                        auxiliary_matrix = np.concatenate((features_new_values_matrix, points), axis=0)
                        features_new_values = np.array(list(itertools.product(*auxiliary_matrix)))

                    # first iteration: new values correspond to the new generated points
                    else:

                        features_new_values = np.array(points).reshape(-1, 1)

                    # add new features to the features_new_values_matrix object
                    features_new_values_matrix = np.append(features_new_values_matrix, points, axis=0)

                    # take original sample features (until the i-th most important)
                    sample_features = np.array([sample[features_reordering[j]] for j in np.arange(i + 1)]).reshape(1, -1)

                    # reorder new features values accordingly to the distance to the original feature values
                    features_new_values = features_new_values[np.argsort(ds.cdist(sample_features, features_new_values,
                                                                                  metric="euclidean")[0]), :]

                    # each values is a list containing the new feature values
                    for u, values in enumerate(features_new_values):

                        # copy the original sample
                        new_sample = np.array(sample).reshape(1, -1)

                        # for each feature until the i-th most important, update the value in new_sample
                        for j, value in enumerate(values):
                            new_sample[0, features_reordering[j]] = value

                        prediction = self.predict(new_sample)

                        # end: pass to the next sample
                        if prediction == contrastive_classes[sample_index]:

                            print("Prediction changed to C2 for feature values %s: end" % (
                                dict(zip(features_reordering[:i + 1], values))))
                            flag = True
                            features_names = np.array([])
                            end_time = time.time()

                            explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d.\n\tPrediction will turn to the class %d by varying " \
                                          "the following features of the following amounts:\n" % \
                                          (ids[sample_index], end_time - start_time, predicted_classes[sample_index],
                                           contrastive_classes[sample_index])

                            for j, value in enumerate(values):
                                explanation += "\t\tPDS band %s channel %s: %+.4f%%\n" % (
                                    DatasetUtils.return_band(features_reordering[j]),
                                    DatasetUtils.return_electrode_number(features_reordering[j]),
                                    ((value - sample[features_reordering[j]]) * 100 / max(
                                        sample[features_reordering[j]], 1e-8)))
                                features_names = np.append(features_names,
                                                           DatasetUtils.return_feature_name(features_reordering[j]))

                            db = DBScan(min_pts=2, dataset=features_names).dbscan()
                            if db.get_n_clusters() >= 1:
                                explanation += "\tClusters.\n" + db.get_clusters()
                            break

                        elif prediction == predicted_classes[sample_index]:

                            print("Prediction is always the same for feature values %s: research continues" % (
                                dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction not changed using given parameters\n" % (
                                    ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True
                                break

                        else:

                            print(
                                "Prediction is changed but not into the desired class for feature values %s: research continues" % (
                                    dict(zip(features_reordering[:i + 1], values))))

                            if i == n_considered_features - 1 and u == features_new_values.shape[0] - 1:
                                end_time = time.time()
                                Plots.plot_feature_importance(most_important_features, most_important_features_abs,
                                                              self.subject_id, ids[sample_index])
                                explanation = "SAMPLE %d (time %.2fs).\n\tActual prediction: %d. Prediction changed but not into the desired class using given parameters\n" % (
                                    ids[sample_index], end_time - start_time, predicted_classes[sample_index])
                                flag = True
                                break
                    if flag:
                        with open(self.base_path + "explanations/" + output_filename, "a", encoding="utf-8") as file_object:
                            file_object.write(explanation)
                            file_object.close()

                        break

        elif self.base_layer_mode == "1vsALL":

            print("No implementation yet")

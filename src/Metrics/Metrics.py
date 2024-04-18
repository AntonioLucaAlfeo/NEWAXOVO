import numpy as np
import math


class Metrics:

    """
      This class implements some static methods useful to compute metrics such as reciprocal_rank,
      mean_reciprocal_rank, average_precision, mean_average_precision, spearman rho distance,
      discounted cumulative gain.
    """

    @classmethod
    def reciprocal_rank(cls, true_order, order, percentage):

        """
          This function computes the Reciprocal Rank. RR is computed as the reciprocal of the index of first
          relevant element in order.

          | Pre-conditions: none.
          | Post-conditions: RR is computed.
          | Main output: Reciprocal Rank.

        :param true_order: an array representing the 'correct' order of the elements.
        :type true_order: ndarray (n_elements,).
        :param order: an array representing the order of the elements.
        :type order: ndarray (n_elements,).
        :param percentage: percentage of elements considered as relevants.
        :type percentage: float (in [0, 1]).
        :return: Reciprocal Rank.
        :rtype: float.
        :raise: none.
        """

        for index in np.arange(order.size):

            if order[index] in true_order[:int(order.size * percentage)]:

                return 1 / (index + 1)

    @classmethod
    def mean_reciprocal_rank(cls, true_order, order, percentage):

        """
          This function computes the Mean Reciprocal Rank (MRR). MRR is computed as the reciprocal of the index of first
          relevant element in order.

          | Pre-conditions: none.
          | Post-conditions: MRR is computed.
          | Main output: Mean Reciprocal Rank.

        :param true_order: a matrix representing the 'correct' order of the elements. Each row represent a different
            query.
        :type true_order: ndarray (n_query, n_elements).
        :param order: a matrix representing the order of the elements to be analyzed. Each row represent a different
            query.
        :type order: ndarray (n_query, n_elements)
        :param percentage: percentage of elements considered as relevants.
        :type percentage: float (in [0, 1]).
        :return: Mean Reciprocal Rank (MRR).
        :rtype: float.
        :raise: none.
        """

        mrr = 0

        for query_index in np.arange(order.shape[0]):

            mrr += cls.reciprocal_rank(true_order[query_index, :], order[query_index, :], percentage)

        return mrr / true_order.shape[0]

    @classmethod
    def average_precision(cls, true_order, order, percentage):

        """
          This function computes the Average Precision. Average Precision is computed as described in
          https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval).

          | Pre-conditions: none.
          | Post-conditions: Average Precision is computed.
          | Main output: Average Precision.

        :param true_order: an array representing the 'correct' order of the elements.
        :type true_order: ndarray (n_elements,).
        :param order: an array representing the order of the elements.
        :type order: ndarray (n_elements,).
        :param percentage: percentage of elements considered as relevants.
        :type percentage: float (in [0, 1]).
        :return: Average Precision.
        :rtype: float.
        :raise: none.
        """

        ap = 0
        relevant_rank = 1

        for i in np.arange(order.size):

            if order[i] not in true_order[:int(order.size * percentage)]: continue

            ap += relevant_rank / (i + 1)
            relevant_rank += 1

        return ap / int(order.size * percentage)

    @classmethod
    def mean_average_precision(cls, true_order, order, percentage):

        """
          This function computes the Average Precision. Average Precision is computed as described in
          https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval).

          | Pre-conditions: none.
          | Post-conditions: Mean Average Precision is computed.
          | Main output: Mean Average Precision.

        :param true_order: a matrix representing the 'correct' order of the elements. Each row represent a different
            query.
        :type true_order: ndarray (n_query, n_elements).
        :param order: a matrix representing the order of the elements to be analyzed. Each row represent a different
            query.
        :type order: ndarray (n_query, n_elements)
        :param percentage: percentage of elements considered as relevants.
        :type percentage: float (in [0, 1]).
        :return: Mean Reciprocal Rank (MRR).
        :rtype: float.
        :raise: none.
        """

        map = 0

        for query_index in np.arange(order.shape[0]):

            map += cls.average_precision(true_order[query_index, :], order[query_index, :], percentage)

        return map / true_order.shape[0]

    @classmethod
    def spearman_rho_distance(cls, true_order, order):

        """
          This function computes the spearman rho distance between true_order and order. It is the square root of the
          sum of the squared differences between the real ranking of an element and the current ranking.

          | Pre-conditions: none.
          | Post-conditions: Spearman Rho Distance is computed.
          | Main output: Spearman Rho Distance.

        :param true_order: an array representing the 'correct' order of the elements.
        :type true_order: ndarray (n_elements,).
        :param order: an array representing the order of the elements.
        :type order: ndarray (n_elements,).
        :return: spearman rho distance between true_order and order.
        :rtype: float.
        :raise: none.
        """

        distance = 0
        true_order = np.ravel(true_order)
        order = np.ravel(order)

        for feature in np.arange(start=1, stop=order.size + 1):

            distance += ((np.where(true_order == feature)[0][0] + 1) - (np.where(order == feature)[0][0] + 1)) ** 2

        return math.sqrt(distance)

    @classmethod
    def discounted_cumulative_gain(cls, true_order, order, percentage):

        """
          This function returns the DCG between true_order and order. See
          https://en.wikipedia.org/wiki/Discounted_cumulative_gain for more info.

          | Pre-conditions: none.
          | Post-conditions: DCG is computed.
          | Main output: Discounted Cumulative Gain.

        :param true_order: an array representing the 'correct' order of the elements.
        :type true_order: ndarray (n_elements,).
        :param order: an array representing the order of the elements.
        :type order: ndarray (n_elements,).
        :param percentage: percentage of elements considered as relevants.
        :type percentage: float (in [0, 1]).
        :return: Discounted Cumulative Gain (DCG).
        :rtype: float.
        :raise: none.
        """

        dcg = 0
        true_order = np.ravel(true_order)
        order = np.ravel(order)

        for index in np.arange(order.size):

            if order[index] not in true_order[:int(order.size * percentage)]: continue

            dcg += 1 / math.log2(index + 2)

        return dcg

    @classmethod
    def percentage_of_informative_at_n(cls, informative_features, order, n):

        """
          This function returns the percentage (in [0, 1]) of informative features contained in the first n
          elements of order.

          | Pre-conditions: none.
          | Post-conditions: Percentage of informative features is computed.
          | Main output: Percentage of informative features.

        :param informative_features: an array containing the list of informative features.
        :type informative_features: ndarray (n_informative,)
        :param order: an array representing the order of the elements.
        :type order: ndarray (n_elements,).
        :param n: number of elements of order to analyze. Only the first n elements of order are considered.
        :type n: int (>0).
        :return: the percentage of informative features contained in the first n elements of order.
        :rtype: float (in [0, 1]).
        :raise: none.
        """

        count = 0
        min_index = min(order.size, n)
        informative_features = np.ravel(informative_features)
        order = np.ravel(order)

        for index in np.arange(min_index):

            if order[index] in informative_features:

                count += 1

        return count / informative_features.size


import numpy as np


class DBScan:

    """
      This class implements the DBScan algorithm. In particular, neighborhood function is designed to work with
      22 electrodes placed as described in https://doi.org/10.3389/fnins.2012.00055.
    """

    def __init__(self, min_pts, dataset):

        """
          This function initialises the DBScan object using the given parameters.

          | Pre-conditions: none.
          | Post-conditions: a new DBScan object is created.
          | Main output: none.

        :param min_pts: minimum number of neighbors that a point must have in order not to be considered as outliers.
        :type min_pts: int (>0).
        :param dataset: features dataset. Each element is the name of a features, such as *theta14* or *gamma11*.
        :type dataset: ndarray (n_elements,).
        :return: none.
        :raise: none.
        """

        self.min_pts = min_pts
        self.visited = [False for _ in dataset]
        self.arranged = [False for _ in dataset]
        self.dataset = dataset

        self.clusters = {}

        self.electrodes_matrix = np.zeros((6, 7), dtype='int')

        self.electrodes_matrix[0, 3] = 1
        self.electrodes_matrix[1, 1] = 2
        self.electrodes_matrix[1, 2] = 3
        self.electrodes_matrix[1, 3] = 4
        self.electrodes_matrix[1, 4] = 5
        self.electrodes_matrix[1, 5] = 6
        self.electrodes_matrix[2, 0] = 7
        self.electrodes_matrix[2, 1] = 8
        self.electrodes_matrix[2, 2] = 9
        self.electrodes_matrix[2, 3] = 10
        self.electrodes_matrix[2, 4] = 11
        self.electrodes_matrix[2, 5] = 12
        self.electrodes_matrix[2, 6] = 13
        self.electrodes_matrix[3, 1] = 14
        self.electrodes_matrix[3, 2] = 15
        self.electrodes_matrix[3, 3] = 16
        self.electrodes_matrix[3, 4] = 17
        self.electrodes_matrix[3, 5] = 18
        self.electrodes_matrix[4, 2] = 19
        self.electrodes_matrix[4, 3] = 20
        self.electrodes_matrix[4, 4] = 21
        self.electrodes_matrix[5, 3] = 22

    def __get_neighbors(self, element):

        """
          This function returns the neighbors of the given element.

          | Pre-conditions: :py:meth:'DBScan.__init__'..
          | Post-conditions: neighbors of the given element are obtained.
          | Main output: an array containing the names of the features considered as neighbors within the dataset.

        :param element: feature name.
        :type element: str.
        :return: names of the names of the features considered as neighbors within the dataset. A feature is considered
            as neighbor of another feature if it comes from an electrode close to the electrode of the other feature.
            An electrode is close to another electrode if there are no other electrodes between them.
        :rtype: ndarray.

        :raise: none.
        """

        electrode_index = int(''.join(filter(str.isdigit, element)))

        current_electrode_position = list(zip(*np.where(self.electrodes_matrix == electrode_index)))[0]

        neighbors_indexes = np.array([self.electrodes_matrix[i][j]
                             for j in np.arange(current_electrode_position[1] - 1, current_electrode_position[1] + 2)
                             for i in np.arange(current_electrode_position[0] - 1, current_electrode_position[0] + 2)
                             if 0 <= i < self.electrodes_matrix.shape[0] and 0 <= j < self.electrodes_matrix.shape[1]])

        neighbors_indexes = np.delete(neighbors_indexes, np.where(neighbors_indexes == 0))
        neighbors_indexes = tuple([str(i).zfill(2) if i < 10 else str(i) for i in neighbors_indexes])
        dataset_without_element = np.delete(self.dataset, np.where(self.dataset == element))

        neighbors = np.array([dataset_without_element[i] for i in np.arange(dataset_without_element.shape[0])
                              if dataset_without_element[i].endswith(neighbors_indexes)])

        return neighbors

    def dbscan(self):

        """
          This function implements the DBScan algorithm.

          | Pre-conditions: :py:meth:'DBScan.__init__'..
          | Post-conditions: clusters of close features are obtained, according to the DBScan algorithm.
          | Main output: none. Clusters are saved in *self.clusters*.

        :return: self.
        :rtype: DBScan.
        :raise: none.
        """

        cluster_index = 0

        for i, feature_name in enumerate(self.dataset):

            if self.visited[i]: continue

            neighbors = self.__get_neighbors(feature_name)

            if neighbors.shape[0] < self.min_pts:

                print("Not enough points in the neighborhood: outlier")

            else:

                cluster_index += 1
                self.clusters[cluster_index] = np.array([])
                self.__expand_cluster(feature_name, neighbors, cluster_index)

        return self

    def __expand_cluster(self, feature_name, neighbors, cluster_index):

        """
          This function expands the cluster whose index is *cluster_index*, starting from *electrode_name* and its
          neighbors.

          | Pre-conditions: :py:meth:'DBScan.__init__'..
          | Post-conditions: new features are added to the cluster whose index is specified by cluster_index.
          | Main output: none. Clusters are saved in *self.clusters*.

        :param feature_name: name of features that will be added to the cluster.
        :type feature_name: str.
        :param neighbors: features close to *electrode_name*.
        :type neighbors: ndarray.
        :param cluster_index: index of the cluster to expand.
        :type cluster_index: unsigned int.
        :return: none.
        :raise: none.
        """

        self.clusters[cluster_index] = np.append(self.clusters[cluster_index], feature_name)
        self.arranged[np.where(self.dataset == feature_name)[0][0]] = True

        while neighbors.size != 0:

            neighbor = neighbors[0]

            neighbor_index = np.where(self.dataset == neighbor)[0][0]

            if not self.visited[neighbor_index]:

                self.visited[neighbor_index] = True

                neighbor_neighbors = self.__get_neighbors(neighbor)

                if neighbor_neighbors.shape[0] >= self.min_pts:

                    neighbors = np.append(neighbors, neighbor_neighbors)

            if not self.arranged[neighbor_index]:

                self.clusters[cluster_index] = np.append(self.clusters[cluster_index], neighbor)
                self.arranged[neighbor_index] = True

            neighbors = np.delete(neighbors, 0)

    def get_clusters(self):

        """
          This function returns a string containing cluster indexes and names formatted in a format suitable for
          printing.

          | Pre-conditions: :py:meth:'DBScan.__init__'..
          | Post-conditions: cluster info are obtained.
          | Main output: a string containing cluster information.

        :return: none.
        :raise: none.
        """

        clusters = ""

        for key, value in self.clusters.items():

            clusters += '\t\t' + str(key) + " : " + str(value) + '\n'

        return clusters

    def get_n_clusters(self):

        """
          This function return the number of clusters found by the algorithm.

          | Pre-conditions: :py:meth:'DBScan.__init__'.
          | Post-conditions: clusters number is obtained. If called before :py:meth:'DBScan.DBScan', it will return 0.
          | Main output: an int representing the number of clusters created.

        :return: n_clusters.
        :rtype: int.
        :raise: none.
        """

        return len(self.clusters)

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity


class DatasetUtils:

    """
      This class implements some static functions useful for dealing with EEG datasets.
    """

    @staticmethod
    def read_dataset(subject_id, n_classes, test_split, dataset_path):

        """
          This function read the dataset whose path is passed as parameter, considering only the specified subject
          and number of classes.

          | Pre-conditions: none.
          | Post-conditions: dataset is read and split into training and test set.
          | Main output: training set and test set, according to the given parameters.

        :param subject_id: the id of the subject.
        :type subject_id: int (between 1 and 9).
        :param n_classes: number of classes considered.
        :type n_classes: int (in {4, 5}).
        :param test_split: the proportion of data that will form the test set.
        :type test_split: float (in [0, 1]).
        :param dataset_path: the path to the dataset.
        :type dataset_path: str.
        :return: training_features, test_features, training_labels, test_labels, training_ids, test_ids.
        :rtype: ndarray, ndarray, ndarray, ndarray, ndarray, ndarray.
        :raise:none.
        """

        dataset = pd.read_csv(dataset_path)
        dataset.drop(dataset[dataset.SUBJECT != subject_id].index,
                     inplace=True)  # consider only subject_id whose identifier is 1
        dataset.drop(["SUBJECT", "RUN"], axis=1, inplace=True)  # drop columns containing subject_id and run info

        if n_classes == 4:
            dataset.drop(dataset[dataset.ACTIVITY == 5].index, inplace=True)  # drop samples related to class 5 (rest)

        x = dataset.loc[:, (dataset.columns != "ACTIVITY") & (dataset.columns != "ID")].to_numpy()
        y = dataset["ACTIVITY"].to_numpy()
        ids = dataset["ID"].to_numpy()

        print("DATASET MADE UP BY %d SAMPLES" % x.shape[0])

        if 0 < test_split < 1:

            training_features, test_features, training_labels, test_labels, training_ids, test_ids = train_test_split(x,
                                                         y, ids, test_size=test_split, stratify=y)

        else:

            test_features = None
            test_labels = None
            test_ids = None
            training_features, training_labels, training_ids = shuffle(x, y, ids)

        training_distribution_dictionary = {}
        test_distribution_dictionary = {}

        for class_index in np.arange(start=1, stop=n_classes + 1):

            training_distribution_dictionary[class_index] = len(np.where(training_labels == class_index)[0])

            if 0 < test_split < 1:
                test_distribution_dictionary[class_index] = len(np.where(test_labels == class_index)[0])

        print("TRAINING SET DISTRIBUTION (%d SAMPLES): %s" % (len(training_labels), training_distribution_dictionary))

        if 0 < test_split < 1:
            print("TEST SET DISTRIBUTION (%d SAMPLES): %s" % (len(test_labels), test_distribution_dictionary))

        return training_features, test_features, training_labels, test_labels, training_ids, test_ids

    @staticmethod
    def read_dataset_by_ids(subject_id, test_ids, dataset_path):

        """
          This function read the dataset whose path is passed as parameter, considering only the specified subject
          and number of classes. Elements whose ids are *test_ids* will represent the test set, while all the other
          samples will represent the training set (5 classes are considered)

          | Pre-conditions: none.
          | Post-conditions: dataset is read and split into training and test set.
          | Main output: training set and test set, according to the given parameters.

        :param subject_id: the id of the subject.
        :type subject_id: int (between 1 and 9).
        :param test_ids: ids of the elements that will form the test set.
        :type test_ids: ndarray (n_samples,)
        :param dataset_path: the path to the dataset.
        :type dataset_path: str.
        :return: training_features, test_features, training_labels, test_labels, training_ids, test_ids.
        :rtype: ndarray, ndarray, ndarray, ndarray, ndarray, ndarray.
        :raise:none.
        """

        dataset = pd.read_csv(dataset_path)
        dataset.drop(dataset[dataset.SUBJECT != subject_id].index,
                     inplace=True)  # consider only subject whose identifier is subject_id
        dataset.drop(["SUBJECT", "RUN"], axis=1, inplace=True)
        test_dataframe = dataset.loc[dataset['ID'].isin(test_ids)]
        training_dataframe = dataset[~dataset.apply(tuple, 1).isin(test_dataframe.apply(tuple, 1))]

        training_features = training_dataframe.loc[:, (training_dataframe.columns != "ACTIVITY") &
                                                      (training_dataframe.columns != "ID")].to_numpy()
        training_labels = training_dataframe["ACTIVITY"].to_numpy()
        training_ids = training_dataframe["ID"].to_numpy()

        test_features = test_dataframe.loc[:, (test_dataframe.columns != "ACTIVITY") &
                                              (test_dataframe.columns != "ID")].to_numpy()
        test_labels = test_dataframe["ACTIVITY"].to_numpy()

        training_features, training_labels, training_ids = shuffle(training_features, training_labels, training_ids)

        return training_features, test_features, training_labels, test_labels, training_ids, test_ids

    @staticmethod
    def get_furthest_elements(subject_id, n_points, dataset_path, n_classes):

        """
          This function considers the portion of the dataset *dataset_path* related to subject *subject_id*
          (*n_classes* are retrieved) and returns the *n_points* furthest elements from the centroid of each
          class. Furthest elements are identified using cosine similarity.

          | Pre-conditions: none.
          | Post-conditions: dataset is read and furthest elements from each centroid are returned.
          | Main output: furthest elements from each class.

        :param subject_id: the id of the subject.
        :type subject_id: int (between 1 and 9).
        :param n_points: number of points returned for each class.
        :type n_points: int (>0).
        :param dataset_path: the path to the dataset.
        :type dataset_path: str.
        :param n_classes: number of classes considered.
        :type n_classes: int (in {4, 5}).
        :return: a matrix containing the furthest *n_points* from the centroid of each class.
        :rtype: ndarray (n_classes, n_points)
        :raise: none.
        """

        dataset_x, _, dataset_y, _, dataset_ids, _ = DatasetUtils.read_dataset(subject_id=subject_id, n_classes=n_classes,
                                                                               test_split=0, dataset_path=dataset_path)

        furthest_elements_ids = np.zeros((n_classes, n_points))

        for label in np.arange(1, n_classes+1):

            relevant_rows = np.where(dataset_y == label)[0]
            training_x = np.array(dataset_x[relevant_rows, :])
            ids = np.array(dataset_ids[relevant_rows])

            centroid = np.mean(training_x, axis=0)

            cosine_similarities = cosine_similarity([centroid], training_x).ravel()

            furthest_points_indexes = np.argsort(cosine_similarities)[:n_points]

            furthest_points_ids = np.array(ids[furthest_points_indexes])

            furthest_elements_ids[label-1, :] = furthest_points_ids

        return furthest_elements_ids

    @staticmethod
    def get_neighbors_elements(subject_id, dataset_path, n_classes):

        """
          This function considers the portion of the dataset *dataset_path* related to subject *subject_id*
          (*n_classes* are retrieved) and for each possible combination of the classes returns a pair of elements
          (one of class_1 and one of class_2) such that no other pair has an higher cosine similarity than it.

          | Pre-conditions: none.
          | Post-conditions: dataset is read and closest elements of different classes are returned.
          | Main output: closest elements of different classes.

        :param subject_id: the id of the subject.
        :type subject_id: int (between 1 and 9).
        :param dataset_path: the path to the dataset.
        :type dataset_path: str.
        :param n_classes: number of classes considered.
        :type n_classes: int (in {4, 5}).
        :return: a matrix containing the furthest *n_points* from the centroid of each class.
        :rtype: ndarray (n_classes, n_points)
        :raise: none.
        """

        dataset_x, _, dataset_y, _, dataset_ids, _ = DatasetUtils.read_dataset(subject_id=subject_id,
                                                    n_classes=n_classes, test_split=0, dataset_path=dataset_path)

        specialist_list = np.array([[i, j] for i in np.arange(start=1, stop=n_classes + 1)
                                    for j in np.arange(start=i + 1, stop=n_classes + 1)])

        ids = np.zeros((len(specialist_list), 2))

        for classifier_index, specialist_classes in enumerate(specialist_list):

            # filtra per coppia di classi
            rows_class1 = np.where(dataset_y == specialist_classes[0])[0]
            rows_class2 = np.where(dataset_y == specialist_classes[1])[0]
            elements_class1 = np.array(dataset_x[rows_class1, :])
            elements_class2 = np.array(dataset_x[rows_class2, :])
            ids_1 = np.array(dataset_ids[rows_class1])
            ids_2 = np.array(dataset_ids[rows_class2])

            cosine_similarities = cosine_similarity(elements_class1, elements_class2)

            max_coordinates = np.unravel_index(cosine_similarities.argmax(), cosine_similarities.shape)

            id_1 = ids_1[max_coordinates[0]]

            id_2 = ids_2[max_coordinates[1]]

            ids[classifier_index, 0] = id_1
            ids[classifier_index, 1] = id_2

        return ids

    @staticmethod
    def add_column_to_csv(filepath, columns_names, columns_values):

        """
          This function add columns *column_names* of values *columns_values* to the file specified in *filepath*.

          | Pre-conditions: none.
          | Post-conditions: new columns are added to the file *filepath*.
          | Main output: none.

        :param filepath: path of file to read. New dataframe is saved into *filepath* too.
        :type filepath: str.
        :param columns_names: an array containing names of the new columns.
        :type columns_names: ndarray (n_newcolumns,)
        :param columns_values: an array or matrix containing values for the new columns.
        :type: columns_values: ndarray (n_samples, n_newcolumns)
        :return: none.
        :raise: none.
        """

        try:

            old_dataframe = pd.read_csv(filepath)

        except (pd.errors.EmptyDataError, FileNotFoundError):

            old_dataframe = pd.DataFrame([])

        new_dataframe = pd.DataFrame(data=columns_values, columns=columns_names)
        new_dataframe = pd.concat([old_dataframe, new_dataframe], axis=1)
        new_dataframe.to_csv(filepath, index=False)

    @staticmethod
    def return_feature_name(feature_number):

        """
          This function returns the name of the feature whose index is passed as parameter.

          | Pre-conditions: none.
          | Post-conditions: name of the feature is obtained.
          | Main output: the name of the feature.

        :param feature_number: the index of the feature for which name is requested.
        :type feature_number: int (between 0 and 131).
        :return: the name of the feature.
        :rtype: str.
        :raise: none.
        """

        features_dictionary = {
            0: "theta01",
            1: "theta02",
            2: "theta03",
            3: "theta04",
            4: "theta05",
            5: "theta06",
            6: "theta07",
            7: "theta08",
            8: "theta09",
            9: "theta10",
            10: "theta11",
            11: "theta12",
            12: "theta13",
            13: "theta14",
            14: "theta15",
            15: "theta16",
            16: "theta17",
            17: "theta18",
            18: "theta19",
            19: "theta20",
            20: "theta21",
            21: "theta22",
            22: "alpha01",
            23: "alpha02",
            24: "alpha03",
            25: "alpha04",
            26: "alpha05",
            27: "alpha06",
            28: "alpha07",
            29: "alpha08",
            30: "alpha09",
            31: "alpha10",
            32: "alpha11",
            33: "alpha12",
            34: "alpha13",
            35: "alpha14",
            36: "alpha15",
            37: "alpha16",
            38: "alpha17",
            39: "alpha18",
            40: "alpha19",
            41: "alpha20",
            42: "alpha21",
            43: "alpha22",
            44: "mu01",
            45: "mu02",
            46: "mu03",
            47: "mu04",
            48: "mu05",
            49: "mu06",
            50: "mu07",
            51: "mu08",
            52: "mu09",
            53: "mu10",
            54: "mu11",
            55: "mu12",
            56: "mu13",
            57: "mu14",
            58: "mu15",
            59: "mu16",
            60: "mu17",
            61: "mu18",
            62: "mu19",
            63: "mu20",
            64: "mu21",
            65: "mu22",
            66: "beta01",
            67: "beta02",
            68: "beta03",
            69: "beta04",
            70: "beta05",
            71: "beta06",
            72: "beta07",
            73: "beta08",
            74: "beta09",
            75: "beta10",
            76: "beta11",
            77: "beta12",
            78: "beta13",
            79: "beta14",
            80: "beta15",
            81: "beta16",
            82: "beta17",
            83: "beta18",
            84: "beta19",
            85: "beta20",
            86: "beta21",
            87: "beta22",
            88: "gamma01",
            89: "gamma02",
            90: "gamma03",
            91: "gamma04",
            92: "gamma05",
            93: "gamma06",
            94: "gamma07",
            95: "gamma08",
            96: "gamma09",
            97: "gamma10",
            98: "gamma11",
            99: "gamma12",
            100: "gamma13",
            101: "gamma14",
            102: "gamma15",
            103: "gamma16",
            104: "gamma17",
            105: "gamma18",
            106: "gamma19",
            107: "gamma20",
            108: "gamma21",
            109: "gamma22",
            110: "alphaMu01",
            111: "alphaMu02",
            112: "alphaMu03",
            113: "alphaMu04",
            114: "alphaMu05",
            115: "alphaMu06",
            116: "alphaMu07",
            117: "alphaMu08",
            118: "alphaMu09",
            119: "alphaMu10",
            120: "alphaMu11",
            121: "alphaMu12",
            122: "alphaMu13",
            123: "alphaMu14",
            124: "alphaMu15",
            125: "alphaMu16",
            126: "alphaMu17",
            127: "alphaMu18",
            128: "alphaMu19",
            129: "alphaMu20",
            130: "alphaMu21",
            131: "alphaMu22"

        }

        return features_dictionary.get(feature_number, lambda: "Invalid configuration name")

    @staticmethod
    def return_electrode_number(feature_number):

        """
          This function returns electrode number of the feature whose index is passed as parameter.

          | Pre-conditions: none.
          | Post-conditions: electrode number of the feature is obtained.
          | Main output: electrode number.

        :param feature_number: the index of the feature for which name is requested.
        :type feature_number: int (between 0 and 131).
        :return: electrode number.
        :rtype: str.
        :raise: none.
        """

        features_dictionary = {
            0: "01",
            1: "02",
            2: "03",
            3: "04",
            4: "05",
            5: "06",
            6: "07",
            7: "08",
            8: "09",
            9: "10",
            10: "11",
            11: "12",
            12: "13",
            13: "14",
            14: "15",
            15: "16",
            16: "17",
            17: "18",
            18: "19",
            19: "20",
            20: "21",
            21: "22",
            22: "01",
            23: "02",
            24: "03",
            25: "04",
            26: "05",
            27: "06",
            28: "07",
            29: "08",
            30: "09",
            31: "10",
            32: "11",
            33: "12",
            34: "13",
            35: "14",
            36: "15",
            37: "16",
            38: "17",
            39: "18",
            40: "19",
            41: "20",
            42: "21",
            43: "22",
            44: "01",
            45: "02",
            46: "03",
            47: "04",
            48: "05",
            49: "06",
            50: "07",
            51: "08",
            52: "09",
            53: "10",
            54: "11",
            55: "12",
            56: "13",
            57: "14",
            58: "15",
            59: "16",
            60: "17",
            61: "18",
            62: "19",
            63: "20",
            64: "21",
            65: "22",
            66: "01",
            67: "02",
            68: "03",
            69: "04",
            70: "05",
            71: "06",
            72: "07",
            73: "08",
            74: "09",
            75: "10",
            76: "11",
            77: "12",
            78: "13",
            79: "14",
            80: "15",
            81: "16",
            82: "17",
            83: "18",
            84: "19",
            85: "20",
            86: "21",
            87: "22",
            88: "01",
            89: "02",
            90: "03",
            91: "04",
            92: "05",
            93: "06",
            94: "07",
            95: "08",
            96: "09",
            97: "10",
            98: "11",
            99: "12",
            100: "13",
            101: "14",
            102: "15",
            103: "16",
            104: "17",
            105: "18",
            106: "19",
            107: "20",
            108: "21",
            109: "22",
            110: "01",
            111: "02",
            112: "03",
            113: "04",
            114: "05",
            115: "06",
            116: "07",
            117: "08",
            118: "09",
            119: "10",
            120: "11",
            121: "12",
            122: "13",
            123: "14",
            124: "15",
            125: "16",
            126: "17",
            127: "18",
            128: "19",
            129: "20",
            130: "21",
            131: "22"

        }

        return features_dictionary.get(feature_number, lambda: "Invalid configuration name")

    @staticmethod
    def return_band(feature_number):

        """
          This function returns band of the feature whose index is passed as parameter.

          | Pre-conditions: none.
          | Post-conditions: band of the feature is obtained.
          | Main output: band.

        :param feature_number: the index of the feature for which name is requested.
        :type feature_number: int (between 0 and 131).
        :return: band name.
        :rtype: str.
        :raise: none.
        """

        if 0 <= feature_number <= 21:

            return "theta"

        elif 22 <= feature_number <= 43:

            return "alpha"

        elif 44 <= feature_number <= 65:

            return "mu"

        elif 66 <= feature_number <= 87:

            return "beta"

        elif 88 <= feature_number <= 109:

            return "gamma"

        elif 110 <= feature_number <= 131:

            return "alphaMu"

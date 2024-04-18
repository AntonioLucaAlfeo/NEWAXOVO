import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from io import BytesIO
from urllib.request import urlopen
from unlzw import unlzw
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


class UCIDatasetLoader:

    """
      This class implements some static methods useful to read UCI datasets from archive repository.
    """

    @classmethod
    def anneal(cls, test_split):

        """
          This function reads the dataset *Anneal* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels. In this dataset, class 4 is empty.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Anneal* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Anneal*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: ANNEAL")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data"
        data = pd.read_csv(url, header=None, na_values="?")
        print("N_INSTANCES: %d" % data.shape[0])
        data = data.fillna(data.mode().iloc[0, :])
        data = data.dropna(how='all', axis=1)  # delete columns that are completely made up by Nan values
        features = SimpleImputer(strategy="most_frequent").fit_transform(data.iloc[:, :-1])
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=0.1,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def autos(cls, test_split):

        """
          This function reads the dataset *Autos* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Autos* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Autos* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        translations_dictionary = {
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'eight': 8,
            'twelve': 12
        }

        print("DATASET: AUTOS")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
        data = pd.read_csv(url, header=None, na_values="?")
        print("N_INSTANCES: %d" % data.shape[0])
        data.iloc[:, 5] = [translations_dictionary[key] for key in data.iloc[:, 5]]
        data.iloc[:, 15] = [translations_dictionary[key] for key in data.iloc[:, 15]]
        features = SimpleImputer(strategy="most_frequent").fit_transform(data.iloc[:, 1:])
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, 0].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def car(cls, test_split):

        """
          This function reads the dataset *Car* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Car* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Car* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: CAR")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1]
        encoder = OrdinalEncoder(categories=[["low", "med", "high", "vhigh"],
                                             ["low", "med", "high", "vhigh"],
                                             ["2", "3", "4", "5more"],
                                             ["2", "4", "more"],
                                             ["small", "med", "big"],
                                             ["low", "med", "high"]])
        features = encoder.fit_transform(features)
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def cleveland(cls, test_split):

        """
          This function reads the dataset *Cleveland* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Cleveland* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Cleveland*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: CLEVELAND")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        data = pd.read_csv(url, header=None, na_values="?")
        data = data.fillna(data.mode().iloc[0, :])
        print("N_INSTANCES: %d" % data.shape[0])
        features = SimpleImputer(strategy="most_frequent").fit_transform(data.iloc[:, :-1].to_numpy())
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def ecoli(cls, test_split):

        """
          This function reads the dataset *Ecoli* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Ecoli* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Ecoli* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: ECOLI")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        data = pd.read_csv(url, header=None, delimiter='\s+')
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def flag(cls, test_split):

        """
          This function reads the dataset *flag* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *flag* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *flag* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: FLAG")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 2:]
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, 1].to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels,  test_labels, classes_number

    @classmethod
    def glass(cls, test_split):

        """
          This function reads the dataset *Glass* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Glass* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Glass* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: GLASS")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def hepatitis_domain(cls, test_split):

        """
          This function reads the dataset *Hepatitis* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels. Contains missing values, denoted by '?'.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Hepatitis* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Hepatitis*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: HEPATITIS")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
        data = pd.read_csv(url, header=None, na_values="?")
        print("N_INSTANCES: %d" % data.shape[0])
        features = SimpleImputer(strategy="most_frequent").fit_transform(data.iloc[:, 1:].to_numpy())
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, 0]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def iris(cls, test_split):

        """
          This function reads the dataset *Iris* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Iris* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Iris* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: IRIS")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1]
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def krkopt(cls, test_split):

        """
          This function reads the dataset *Krkopt* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Krkopt* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Krkopt*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: KRKOPT")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1]
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def letter(cls, test_split):

        """
          This function reads the dataset *Letter* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Letter* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Letter*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: LETTER")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:]
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, 0]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def pageblocks(cls, test_split):

        """
          This function reads the dataset *Pageblocks* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Pageblocks* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Pageblocks*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: PAGEBLOCKS")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.data.Z"
        compressed_file = urlopen(url).read()
        uncompressed_file = unlzw(compressed_file)
        data = pd.read_csv(BytesIO(uncompressed_file), header=None, delimiter='\s+')
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def penbased(cls, test_split):

        """
          This function reads the dataset *Penbased* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Penbased* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Penbased*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: PENBASED")

        url_training = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
        url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes"
        data_training = pd.read_csv(url_training, header=None)
        data_test = pd.read_csv(url_test, header=None)
        data = data_training.append(data_test, ignore_index=True)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def satimage(cls, test_split):

        """
          This function reads the dataset *Satimage* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Satimage* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Satimage*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        url_training = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"
        url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"
        data_training = pd.read_csv(url_training, header=None, delimiter='\s+')
        data_test = pd.read_csv(url_test, header=None, delimiter='\s+')
        data = data_training.append(data_test, ignore_index=True)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def segment(cls, test_split):

        """
          This function reads the dataset *Segment* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Segment* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Segment*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: SEGMENT")

        url_training = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
        url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test"
        data_training = pd.read_csv(url_training, header=None, skiprows=5)
        data_test = pd.read_csv(url_test, header=None, skiprows=5)
        data = data_training.append(data_test, ignore_index=True)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, 0].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def shuttle(cls, test_split):

        """
          This function reads the dataset *Shuttle* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Shuttle* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Shuttle*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: SHUTTLE")

        url_training = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z"
        url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst"
        compressed_file = urlopen(url_training).read()
        uncompressed_file = unlzw(compressed_file)
        data_training = pd.read_csv(BytesIO(uncompressed_file), header=None, delimiter='\s+', names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "C"])
        data_test = pd.read_csv(url_test, header=None, delimiter='\s+', names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "C"])
        data = data_training.append(data_test, ignore_index=True)
        data.drop(data[data.C == 7].index, inplace=True)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].to_numpy()
        classes_number = len(np.unique(labels))
        print(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def splice(cls, test_split):

        """
          This function reads the dataset *Splice* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Splice* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Splice*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: SPLICE")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        data = data.drop(columns=1)
        labels = data.iloc[:, 0]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        features = data.iloc[:, -1]
        features = features.str.strip()
        features = features.apply(lambda x: pd.Series(list(x)))
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def vehicle(cls, test_split):

        """
          This function reads the dataset *Vehicle* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Vehicle* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Vehicle*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: VEHICLE")

        url_a = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat"
        url_b = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat"
        url_c = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat"
        url_d = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat"
        url_e = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat"
        url_f = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat"
        url_g = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat"
        url_h = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat"
        url_i = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat"
        data_a = pd.read_csv(url_a, header=None, delimiter='\s+')
        data_b = pd.read_csv(url_b, header=None, delimiter='\s+')
        data_c = pd.read_csv(url_c, header=None, delimiter='\s+')
        data_d = pd.read_csv(url_d, header=None, delimiter='\s+')
        data_e = pd.read_csv(url_e, header=None, delimiter='\s+')
        data_f = pd.read_csv(url_f, header=None, delimiter='\s+')
        data_g = pd.read_csv(url_g, header=None, delimiter='\s+')
        data_h = pd.read_csv(url_h, header=None, delimiter='\s+')
        data_i = pd.read_csv(url_i, header=None, delimiter='\s+')
        data = pd.concat([data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h, data_i], ignore_index=True)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:

            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def vowel(cls, test_split):

        """
          This function reads the dataset *Vowel* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Vowel* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Vowel* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: VOWEL")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data"
        data = pd.read_csv(url, header=None, delimiter='\s+')
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def waveform(cls, test_split):

        """
          This function reads the dataset *Waveform* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Waveform* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Waveform*
            dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: WAVEFORM")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform-+noise.data.Z"
        compressed_file = urlopen(url).read()
        uncompressed_file = unlzw(compressed_file)
        data = pd.read_csv(BytesIO(uncompressed_file), header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, :-1]
        print("N_FEATURES: %d" % features.shape[1])
        features = pd.get_dummies(features).to_numpy()
        labels = data.iloc[:, -1]
        labels = labels.astype('category').cat.codes + 1
        labels = labels.to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)
        cls.print_class_distribution(labels)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def yeast(cls, test_split):

        """
          This function reads the dataset *Yeast* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Yeast* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Yeast* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: YEAST")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        data = pd.read_csv(url, header=None, delimiter='\s+')
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].astype('category').cat.codes.to_numpy() + 1
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def zoo(cls, test_split):

        """
          This function reads the dataset *Zoo* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Zoo* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *Zoo* dataset.

        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        print("DATASET: ZOO")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
        data = pd.read_csv(url, header=None)
        print("N_INSTANCES: %d" % data.shape[0])
        features = data.iloc[:, 1:-1].to_numpy()
        print("N_FEATURES: %d" % features.shape[1])
        labels = data.iloc[:, -1].to_numpy()
        classes_number = len(np.unique(labels))
        print("N_CLASSES: %d" % classes_number)

        if test_split == 0:
            features, labels = shuffle(features, labels)
            return features, None, labels, None, classes_number

        training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                                          test_size=test_split,
                                                                                          stratify=labels)
        return training_features, test_features, training_labels, test_labels, classes_number

    @classmethod
    def uci_dataset_handler(cls, dataset_name, test_split):

        """
          This function reads the dataset *dataset_name* from the repository and returns a training set and test set
          accordingly to the proportion specified in *test_split*. Training and test set are split in a stratified
          random way. Categorical features (if present) are converted in numerical features. Class labels are
          converted in numerical labels.

          | Pre-conditions: none.
          | Post-conditions: training and test set obtained from dataset *Vehicle* are returned.
          | Main output: training features, test_features, training labels, test_labels, n_classes from *dataset_name*
            dataset.

        :param dataset_name: name of the dataset.
        :type dataset_name: str. One between *anneal*, *autos*, *car*, *cleveland*, *ecoli*, *flag*, *glass*,
            *hepatitis*, *iris*, *krkopt*, *letter*, *pageblocks*, *penbased*, *satimage*, *shuttle*, *segment*,
            *splice*, *vehicle*, *vowel*, *waveform*, *yeast*, *zoo*.
        :param test_split: Percentage of dataset considered as test set.
        :type test_split: float (in [0, 1]).
        :return: training features, test_features, training labels, test_labels, n_classes.
        :rtype: ndarray, ndarray, ndarray, ndarray, int.
        :raise: none.
        """

        switcher = {
            "anneal": cls.anneal,
            "autos": cls.autos,
            "car": cls.car,
            "cleveland": cls.cleveland,
            "ecoli": cls.ecoli,
            "flag": cls.flag,
            "glass": cls.glass,
            "hepatitis": cls.hepatitis_domain,
            "iris": cls.iris,
            "krkopt": cls.krkopt,
            "letter": cls.letter,
            "pageblocks": cls.pageblocks,
            "penbased": cls.penbased,
            "satimage": cls.satimage,
            "shuttle": cls.shuttle,
            "segment": cls.segment,
            "splice": cls.splice,
            "vehicle": cls.vehicle,
            "vowel": cls.vowel,
            "waveform": cls.waveform,
            "yeast": cls.yeast,
            "zoo": cls.zoo
        }

        func = switcher.get(dataset_name, lambda: "Invalid dataset name")
        return func(test_split)

    @classmethod
    def print_class_distribution(cls, labels):

        """
          This function returns the distribution of the classes of the dataset whose labels are passed in *labels*.

          | Pre-conditions: none.
          | Post-conditions: distribution of the dataset is obtained.
          | Main output: dictionary containing class labels and their number of occurrences.

        :param labels: an array representing the labels of all the samples of the dataset.
        :type labels: ndarray (n_samples,).
        :return: none,
        :raise: none.
        """

        distribution_dictionary = {}

        for class_index in np.arange(start=1, stop=len(np.unique(labels))+1):

            distribution_dictionary[class_index] = len(np.where(labels == class_index)[0])

        print(distribution_dictionary)

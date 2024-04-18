import numpy as np
from src.NAWAX.NAWAX import NAWAX
from src.Utils.DatasetUtils import DatasetUtils

if __name__ == "__main__":

    base_path = "C:/Users/Luca/Desktop/NEWAXOVO/"
    dataset_path = base_path + "data/PDSsPerBand_MinMax_w4s_step05s_ids.csv"
    subject_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = 5
    n_neighbors = 3

    for subject_id in subject_ids:

        test_ids = DatasetUtils.get_neighbors_elements(subject_id=subject_id, n_classes=n_classes,
                                         dataset_path=dataset_path).ravel()

        training_x, test_x, training_y, test_y, training_ids, test_ids = DatasetUtils.read_dataset_by_ids(
            subject_id=subject_id, test_ids=test_ids, dataset_path=dataset_path)

        nawax = NAWAX(n_classes=n_classes, subject_id=subject_id, n_neighbors=n_neighbors,
                      base_layer_mode="1vs1", samples_balancing="none").fit(training_x, training_y)

        nawax.explain(x=test_x, ids=test_ids, mode="all_classes", q=5, k=2, p=0.09)




from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.DRCW.DRCW import DRCW
from src.NAWAX.NAWAX import NAWAX
from src.TAWAX.TAWAX import TAWAX
import numpy as np
from src.Utils.DatasetUtils import DatasetUtils

if __name__ == "__main__":

    subject_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    base_path = "C:/Users/Luca/Desktop/NEWAXOVO/"
    dataset = base_path + "data/PDSsPerBand_MinMax_w4s_step05s_ids.csv"
    n_classes = 4
    n_neighbors = 3
    balancing = "none"
    configuration = "1vs1"
    test_split = 0

    for subject_id in subject_ids:

        x, _, y, _, ids, _ = DatasetUtils.read_dataset(subject_id=subject_id,
                                                       n_classes=n_classes,
                                                       test_split=test_split,
                                                       dataset_path=dataset)

        kf = StratifiedKFold(n_splits=10)
        predictions_nawax = np.empty(x.shape[0])
        predictions_drcw = np.empty(x.shape[0])

        for [training_indexes, test_indexes] in kf.split(x, y):

            nawax = NAWAX(n_classes=n_classes, subject_id=subject_id, n_neighbors=n_neighbors,
                      base_layer_mode=configuration,
                      samples_balancing=balancing).fit(x[training_indexes], y[training_indexes])

            drcwovo = DRCW(n_classes=n_classes, subject_id=subject_id, n_neighbors=n_neighbors,
                           samples_balancing=balancing).fit(x[training_indexes], y[training_indexes])

            tawaxovo = TAWAX(n_classes=n_classes, subject_id=subject_id,
                        samples_balancing=balancing).fit(x[training_indexes], y[training_indexes])

            y_pred_test_nawax = nawax.processing_history_predict(x[test_indexes], y[test_indexes], ids[test_indexes])
            y_pred_test_drcw = drcwovo.processing_history_predict(x[test_indexes], y[test_indexes], ids[test_indexes])
            tawaxovo.processing_history_predict(x[test_indexes], y[test_indexes], ids[test_indexes])

            predictions_nawax[test_indexes] = y_pred_test_nawax
            predictions_drcw[test_indexes] = y_pred_test_drcw

        print("SUBJECT %d NAWAX ACCURACY %.4f" % (subject_id, accuracy_score(y, predictions_nawax)))
        print("SUBJECT %d DRCW ACCURACY %.4f" % (subject_id, accuracy_score(y, predictions_drcw)))
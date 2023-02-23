# From official organiser of ICFHR 2020 Competition
# https://github.com/anguelos/wi19_evaluate

import unittest

import numpy as np

from utils.wi19_evaluate import get_sorted_retrievals, get_precision_recall_matrices, compute_map, compute_fscore, \
    compute_roc


class TestMetrics(unittest.TestCase):
    def test_sorted_retrievals(self):
        in_classes = np.array([0, 0, 1, 1, 2, 0], dtype="int64")
        in_D = np.array([[.0, .1, .2, .9, .4, .5],
                         [.1, .0, .3, .4, .5, .6],
                         [.2, .3, .0, .5, .6, .7],
                         [.9, .4, .5, .0, .7, .8],
                         [.4, .5, .6, .7, .0, .9],
                         [.5, .6, .7, .8, .9, .0]], dtype="float")
        target_sorted_retrievals = np.array([[1, 1, 0, 0, 1, 0],
                                             [1, 1, 0, 0, 0, 1],
                                             [1, 0, 0, 1, 0, 0],
                                             [1, 0, 1, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 0, 0, 0]], dtype=bool)

        sorted_retrievals = get_sorted_retrievals(
            in_D, in_classes, remove_self_column=False)
        np.testing.assert_array_equal(
            sorted_retrievals,
            target_sorted_retrievals,
            err_msg='Sorted Retrievals wrong.',
            verbose=True)
        # Test remove self column
        sorted_retrievals = get_sorted_retrievals(in_D, in_classes)
        np.testing.assert_array_equal(sorted_retrievals,
                                      target_sorted_retrievals[:,
                                      1:],
                                      err_msg='Sorted Retrievals wrong.',
                                      verbose=True)

    def test_precision_recall(self):
        in_classes = np.array([0, 0, 1, 1, 2, 0], dtype="int64")
        in_D = np.array([[.0, .1, .2, .9, .4, .5],
                         [.1, .0, .3, .4, .5, .6],
                         [.2, .3, .0, .5, .6, .7],
                         [.9, .4, .5, .0, .7, .8],
                         [.4, .5, .6, .7, .0, .9],
                         [.5, .6, .7, .8, .9, .0]], dtype="float")

        target_pr = np.array([[1.0 /
                               1, 2.0 /
                               2, 2.0 /
                               3, 2.0 /
                               4, 3.0 /
                               5, 3.0 /
                               6], [1.0 /
                                    1, 2.0 /
                                    2, 2.0 /
                                    3, 2.0 /
                                    4, 2.0 /
                                    5, 3.0 /
                                    6], [1.0 /
                                         1, 1.0 /
                                         2, 1.0 /
                                         3, 2.0 /
                                         4, 2.0 /
                                         5, 2.0 /
                                         6], [1.0 /
                                              1, 1.0 /
                                              2, 2.0 /
                                              3, 2.0 /
                                              4, 2.0 /
                                              5, 2.0 /
                                              6], [1.0 /
                                                   1, 1.0 /
                                                   2, 1.0 /
                                                   3, 1.0 /
                                                   4, 1.0 /
                                                   5, 1.0 /
                                                   6], [1.0 /
                                                        1, 2.0 /
                                                        2, 3.0 /
                                                        3, 3.0 /
                                                        4, 3.0 /
                                                        5, 3.0 /
                                                        6]])

        target_rec = np.array([[1.0 /
                                3, 2.0 /
                                3, 2.0 /
                                3, 2.0 /
                                3, 3.0 /
                                3, 3.0 /
                                3], [1.0 /
                                     3, 2.0 /
                                     3, 2.0 /
                                     3, 2.0 /
                                     3, 2.0 /
                                     3, 3.0 /
                                     3], [1.0 /
                                          2, 1.0 /
                                          2, 1.0 /
                                          2, 2.0 /
                                          2, 2.0 /
                                          2, 2.0 /
                                          2], [1.0 /
                                               2, 1.0 /
                                               2, 2.0 /
                                               2, 2.0 /
                                               2, 2.0 /
                                               2, 2.0 /
                                               2], [1.0 /
                                                    1, 1.0 /
                                                    1, 1.0 /
                                                    1, 1.0 /
                                                    1, 1.0 /
                                                    1, 1.0 /
                                                    1], [1.0 /
                                                         3, 2.0 /
                                                         3, 3.0 /
                                                         3, 3.0 /
                                                         3, 3.0 /
                                                         3, 3.0 /
                                                         3]])

        pr, rec, _ = get_precision_recall_matrices(
            in_D, in_classes, remove_self_column=False)

        np.testing.assert_array_equal(
            pr, target_pr, err_msg='Precision @ wrong.', verbose=True)
        np.testing.assert_array_equal(
            rec, target_rec, err_msg='Recall @ wrong.', verbose=True)

    def test_compute_map(self):
        in_sorted_retrievals = np.array([[1, 1, 0, 0, 1, 0],
                                         [1, 1, 0, 0, 0, 1],
                                         [1, 0, 0, 1, 0, 0],
                                         [1, 0, 1, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0],
                                         [1, 1, 1, 0, 0, 0]], dtype=bool)
        in_pr = np.array([[1.0 /
                           1, 2.0 /
                           2, 2.0 /
                           3, 2.0 /
                           4, 3.0 /
                           5, 3.0 /
                           6], [1.0 /
                                1, 2.0 /
                                2, 2.0 /
                                3, 2.0 /
                                4, 2.0 /
                                5, 3.0 /
                                6], [1.0 /
                                     1, 1.0 /
                                     2, 1.0 /
                                     3, 2.0 /
                                     4, 2.0 /
                                     5, 2.0 /
                                     6], [1.0 /
                                          1, 1.0 /
                                          2, 2.0 /
                                          3, 2.0 /
                                          4, 2.0 /
                                          5, 2.0 /
                                          6], [1.0 /
                                               1, 1.0 /
                                               2, 1.0 /
                                               3, 1.0 /
                                               4, 1.0 /
                                               5, 1.0 /
                                               6], [1.0 /
                                                    1, 2.0 /
                                                    2, 3.0 /
                                                    3, 3.0 /
                                                    4, 3.0 /
                                                    5, 3.0 /
                                                    6]])

        target_map = ((1.0 + 1.0 + 3.0 / 5) / 3 + (1 + 1 + 3.0 / 6) / 3 +
                      (1 + 2.0 / 4) / 2 + (1 + 2.0 / 3) / 2 + 1 + (1 + 1 + 1.0) / 3) / 6

        mAP = compute_map(in_pr, in_sorted_retrievals)
        np.testing.assert_equal(mAP, target_map)

    def test_compute_fscore(self):
        in_sorted_retrievals = np.array([[1, 0, 0, 1, 0],  # 1
                                         [1, 0, 0, 0, 1],  # 1
                                         [0, 0, 1, 0, 0],  # 1
                                         [0, 1, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0],  # 0
                                         [1, 1, 0, 0, 0]], dtype=bool)
        in_relevant_estimate = np.array([1, 2, 3, 1, 0, 2], dtype="int64")
        tp = float(1 + 1 + 1 + 0 + 0 + 2)
        relevant = in_sorted_retrievals.sum()
        retrieved = in_relevant_estimate.sum()
        target_precision = tp / retrieved
        target_recall = tp / relevant
        target_fscore = 2 * target_precision * \
                        target_recall / (target_precision + target_recall)
        fscore, precision, recall = compute_fscore(
            in_sorted_retrievals, in_relevant_estimate)
        np.testing.assert_equal(precision, target_precision)
        np.testing.assert_equal(recall, target_recall)
        np.testing.assert_equal(fscore, target_fscore)

    def test_compute_roc(self):
        in_sorted_retrievals = np.array([[1, 0, 0, 1, 0],  # 1
                                         [1, 0, 0, 0, 1],  # 1
                                         [0, 0, 1, 0, 0],  # 1
                                         [0, 1, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0],  # 0
                                         [1, 1, 0, 0, 0]], dtype=bool)
        target_tp = np.array([3, 5, 6, 7, 8], dtype="float")
        target_retrieved = np.array([6, 12, 18, 24, 30], dtype="float")
        target_relevant = np.array([8, 8, 8, 8, 8], dtype="float")
        # TODO (anguelos) replace precison with fallout in the test
        target_precision = target_tp / target_retrieved
        target_recall = target_tp / target_relevant
        target_roc = {"fallout": target_precision, "recall": target_recall}
        roc = compute_roc(in_sorted_retrievals)
        self.assertListEqual(sorted(roc.keys()), sorted(target_roc.keys()))

        # np.testing.assert_equal(roc["fallout"], target_roc["fallout"])
        np.testing.assert_equal(roc["recall"], target_roc["recall"])


if __name__ == '__main__':
    unittest.main()

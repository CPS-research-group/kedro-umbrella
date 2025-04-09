from kedro_umbrella.library import *
import numpy as np


def test_difference_metric():
    def do_test(grid, ex_classif_diff, ex_regr_diff, ex_regr_pos_diff, step=1):
        # Test classification difference metric
        classif_diff = difference_metric(grid, diff_type="classification", step=step)
        assert np.array_equal(
            classif_diff[0], ex_classif_diff
        ), "Classification difference metric failed"

        # Test regression difference metric
        diff, diff_pos = difference_metric(grid, diff_type="regression", step=step)

        shape_grid = tuple(np.array(diff.shape) + 2 * step)
        assert grid.shape == shape_grid

        assert np.array_equal(diff, ex_regr_diff), "Regression difference metric failed"
        assert np.array_equal(
            diff_pos, ex_regr_pos_diff
        ), "Regression diff position failed"

    def test_1():
        # Create a sample grid
        grid = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ]
        )

        # Expected output for classification difference metric
        ex_classif_diff = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        # Expected output for regression difference metric
        ex_regr_diff = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        ex_regr_pos_diff = np.array(
            [
                [(0, 0), (0, 1), (0, 2)],
                [(1, 0), (1, 1), (1, 2)],
                [(2, 0), (2, 1), (2, 2)],
            ]
        )
        do_test(grid, ex_classif_diff, ex_regr_diff, ex_regr_pos_diff)

    def test_2():
        # Create a sample grid
        grid = np.ones((5, 5))

        # Expected output for classification difference metric
        ex_classif_diff = np.zeros((3, 3))

        # Expected output for regression difference metric
        ex_regr_diff = np.zeros((3, 3))
        ex_regr_pos_diff = np.array(
            [
                [(0, 1), (0, 2), (0, 3)],
                [(1, 1), (1, 2), (1, 3)],
                [(2, 1), (2, 2), (2, 3)],
            ]
        )
        do_test(grid, ex_classif_diff, ex_regr_diff, ex_regr_pos_diff)

    def test_3():
        # Create a sample grid
        grid = np.array(
            #    0    1   2   3   4  5
            [
                [1, 2, 3, 4, 5, 6],  # 0
                [7, 8, 9, 10, 11, 12],  # 1
                [13, 14, 15, 16, 17, 18],  # 2
                [19, 20, 21, 22, 23, 24],  # 3
                [25, 26, 27, 28, 29, 30],  # 4
                [31, 32, 33, 34, 35, 36],  # 5
            ]
        )

        # Expected output for classification difference metric
        ex_classif_diff = np.array([[1, 1], [1, 1]])

        # Expected output for regression difference metric
        ex_regr_diff = np.array([[14, 14], [14, 14]])
        ex_regr_pos_diff = np.array(
            [
                [(0, 0), (0, 1)],
                [(1, 0), (1, 1)],
            ]
        )
        do_test(grid, ex_classif_diff, ex_regr_diff, ex_regr_pos_diff, step=2)

    test_1()
    test_2()
    test_3()


if __name__ == "__main__":
    test_difference_metric()
    print("All tests passed.")

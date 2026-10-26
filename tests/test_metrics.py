"""Metrics checked against numbers worked out by hand.

Example used throughout (3 classes, 10 test windows):

                 predicted
               A    B    C
    true  A    3    1    0      (4 windows were really A)
          B    0    2    1      (3 were B)
          C    1    0    2      (3 were C)

    accuracy = (3+2+2)/10 = 0.7
    class A: precision 3/4 = 0.75, recall 3/4 = 0.75, F1 = 0.75
    class B: precision 2/3,        recall 2/3,        F1 = 2/3
    class C: precision 2/3,        recall 2/3,        F1 = 2/3
    macro-F1 = (0.75 + 2/3 + 2/3) / 3 = 0.694444...
    chance (always guess the most common true class, A) = 4/10 = 0.4
"""

import numpy as np
import pytest

from emgdec import metrics as M

CM = np.array([[3, 1, 0], [0, 2, 1], [1, 0, 2]])


def test_confusion_counts_pairs():
    y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 0, 2, 2]
    assert np.array_equal(M.confusion(y_true, y_pred, 3), CM)


def test_confusion_rejects_bad_labels():
    with pytest.raises(ValueError):
        M.confusion([0, 3], [0, 0], 3)
    with pytest.raises(ValueError):
        M.confusion([0, 1], [0], 3)


def test_accuracy_by_hand():
    assert M.accuracy(CM) == pytest.approx(0.7)


def test_per_class_f1_by_hand():
    f1 = M.per_class_f1(CM)
    assert f1 == pytest.approx([0.75, 2 / 3, 2 / 3])


def test_macro_f1_by_hand():
    assert M.macro_f1(CM) == pytest.approx((0.75 + 2 / 3 + 2 / 3) / 3)


def test_chance_by_hand():
    assert M.chance_accuracy(CM) == pytest.approx(0.4)


def test_absent_class_is_ignored_in_macro_f1():
    # Class C never occurs in the test set (empty row), so it must not count.
    # A: tp 3, predicted 3, actually 4 → p=1.0, r=0.75, F1 = 2·1·0.75/1.75 = 0.857142...
    # B: tp 2, predicted 3, actually 3 → p=2/3, r=2/3, F1 = 2/3
    cm = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 0]])
    f1 = M.per_class_f1(cm)
    assert np.isnan(f1[2])
    assert f1[0] == pytest.approx(2 * 1.0 * 0.75 / 1.75)
    assert f1[1] == pytest.approx(2 / 3)
    assert M.macro_f1(cm) == pytest.approx((2 * 1.0 * 0.75 / 1.75 + 2 / 3) / 2)


def test_perfect_and_empty():
    assert M.accuracy(np.eye(4, dtype=int) * 5) == 1.0
    assert M.macro_f1(np.eye(4, dtype=int) * 5) == 1.0
    assert np.isnan(M.accuracy(np.zeros((3, 3), dtype=int)))

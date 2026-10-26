"""The aggregator must reproduce hand-computed means and refuse malformed runs."""

import numpy as np
import pytest

from emgdec import evaluate as E
from emgdec.config import N_GESTURES


def _rec(name, correct, wrong, **kw):
    """A 17-class confusion matrix with `correct` on the diagonal of class 0 and `wrong` at (0,1)."""
    cm = np.zeros((N_GESTURES, N_GESTURES), dtype=int)
    cm[0, 0] = correct
    cm[0, 1] = wrong
    return E.RunRecord(
        model=kw.get("model", "lda"), config=kw.get("config", "both"), kind=kw.get("kind", "cross_day"),
        name=name, confusion=cm.tolist(), n_train=100, n_val=10, n_test=correct + wrong,
        train_seconds=1.5, seed=1, params={"shrinkage": "auto"},
    )


def test_roundtrip_and_means(tmp_path):
    E.save_run(_rec("s01", 8, 2), tmp_path)  # accuracy 0.8
    E.save_run(_rec("s02", 6, 4), tmp_path)  # accuracy 0.6
    res = E.build(tmp_path)
    assert res["n_runs"] == 2
    cell = res["cells"]["lda"]["both"]["cross_day"]
    assert cell["n_splits"] == 2
    assert cell["accuracy_mean"] == pytest.approx(0.7)
    assert cell["accuracy_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
    assert cell["n_test_windows"] == 20
    assert cell["confusion"][0][0] == 14 and cell["confusion"][0][1] == 6
    assert cell["per_split"]["s01"]["accuracy"] == pytest.approx(0.8)
    assert cell["train_seconds_total"] == pytest.approx(3.0)
    assert len(res["table"]) == 1 and res["table"][0]["kind"] == "cross_day"


def test_refuses_wrong_shape_or_total(tmp_path):
    bad = _rec("x", 5, 5)
    bad.confusion = [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        E.save_run(bad, tmp_path)
    bad2 = _rec("y", 5, 5)
    bad2.n_test = 99
    with pytest.raises(ValueError):
        E.save_run(bad2, tmp_path)
    bad3 = _rec("z", 5, 5, model="svm")
    with pytest.raises(ValueError):
        E.save_run(bad3, tmp_path)


def test_cells_are_separated_by_model_config_kind(tmp_path):
    E.save_run(_rec("a", 9, 1, model="lda", config="wrist", kind="within_subject"), tmp_path)
    E.save_run(_rec("a", 5, 5, model="cnn", config="wrist", kind="within_subject"), tmp_path)
    res = E.build(tmp_path)
    assert res["cells"]["lda"]["wrist"]["within_subject"]["accuracy_mean"] == pytest.approx(0.9)
    assert res["cells"]["cnn"]["wrist"]["within_subject"]["accuracy_mean"] == pytest.approx(0.5)

import numpy as np
import pytest

from segbench.metrics import SegmentationMetrics, confusion_matrix, metrics_from_confusion


def test_prediction_parfaite():
    target = np.array([0, 0, 1, 2, 2, 2])
    m = metrics_from_confusion(confusion_matrix(target, target, num_classes=3))
    assert m["oa"] == 1.0
    assert m["miou"] == 1.0
    assert m["macc"] == 1.0


def test_valeurs_calculees_a_la_main():
    target = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    m = metrics_from_confusion(confusion_matrix(pred, target, num_classes=2))
    # classe 0 : TP=1, FN=1, FP=0 -> IoU 1/2 ; classe 1 : TP=2, FP=1 -> IoU 2/3
    np.testing.assert_allclose(m["iou"], [1 / 2, 2 / 3])
    np.testing.assert_allclose(m["acc"], [1 / 2, 1.0])
    assert m["oa"] == pytest.approx(3 / 4)
    assert m["miou"] == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_ignore_index_exclu():
    target = np.array([0, -1, 1])
    pred = np.array([0, 1, 1])
    cm = confusion_matrix(pred, target, num_classes=2, ignore_index=-1)
    assert cm.sum() == 2
    assert metrics_from_confusion(cm)["oa"] == 1.0


def test_classe_absente_exclue_de_la_moyenne():
    target = np.array([0, 0, 1])
    m = metrics_from_confusion(confusion_matrix(target, target, num_classes=3))
    assert np.isnan(m["iou"][2])
    assert m["miou"] == 1.0


def test_accumulation_sur_plusieurs_scenes():
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(np.array([0, 1]), np.array([0, 0]))
    metrics.update(np.array([1, 1]), np.array([1, 1]))
    direct = confusion_matrix(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]), num_classes=2)
    np.testing.assert_array_equal(metrics.cm, direct)
    assert "mIoU" in metrics.summary()


def test_etiquette_hors_bornes():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0]), np.array([5]), num_classes=2)

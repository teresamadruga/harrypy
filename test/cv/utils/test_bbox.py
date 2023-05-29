import numpy as np
import pytest

from cv.utils.bbox import BoundingBox


def test_bounding_box_creation():
    bbox = BoundingBox(0, 0, 100, 100, 200, 200)
    assert np.array_equal(bbox(), np.array([0, 0, 100, 100]))
    assert np.array_equal(bbox.image_dim, np.array([200, 200]))


def test_bounding_box_invalid_type_input():
    with pytest.raises(TypeError):
        bbox = BoundingBox(0, 0, 100, 100, "200", 200)


def test_bounding_box_invalid_value_input():
    with pytest.raises(ValueError):
        bbox = BoundingBox(-1, 0, 100, 100, 200, 200)
        # bbox = BoundingBox(0, -1, 100, 100, 200, 200)
        # bbox = BoundingBox(0, 0, 100, 300, 200, 200)
        # bbox = BoundingBox(0, 0, 100, 100, 50, 200)


def test_bounding_box_call():
    bbox = BoundingBox(0, 0, 100, 100, 200, 200)
    assert np.array_equal(bbox(), np.array([0, 0, 100, 100]))


def test_bounding_box_area():
    bbox = BoundingBox(0, 0, 100, 100, 200, 200)
    assert bbox.area == 10000


def test_bounding_box_aspect_ratio():
    bbox = BoundingBox(0, 0, 100, 200, 200, 400)
    assert bbox.aspect_ratio == 2


def test_bounding_box_iou():
    bbox1 = BoundingBox(0, 0, 100, 100, 200, 200)
    bbox2 = BoundingBox(50, 50, 150, 150, 200, 200)
    assert bbox1.iou(bbox2.bbox) == pytest.approx(0.14285714)
    bbox1 = BoundingBox(10, 10, 50, 50, 100, 100)
    bbox2 = BoundingBox(30, 30, 70, 70, 100, 100)
    assert bbox1.iou(bbox2.bbox) == pytest.approx(0.14285714285714285)

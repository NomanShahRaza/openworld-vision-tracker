import numpy as np

from app.visualization import mask_to_polygon


def test_mask_to_polygon_returns_points():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:40, 15:45] = True
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    assert len(polygon) >= 4

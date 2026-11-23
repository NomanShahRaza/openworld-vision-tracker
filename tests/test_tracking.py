import numpy as np

from app.tracking import BotSort, box_iou, cosine_similarity


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values)


def test_box_iou():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([5, 5, 15, 15], dtype=np.float32)
    assert round(box_iou(a, b), 4) == 0.1429


def test_cosine_similarity():
    assert cosine_similarity(unit([1, 0]), unit([1, 0])) > 0.999
    assert abs(cosine_similarity(unit([1, 0]), unit([0, 1]))) < 1e-6


def test_botsort_keeps_identity_with_reid_and_motion():
    tracker = BotSort(max_age=2, use_cmc=False)
    embedding = unit([1.0, 0.1, 0.0])

    ids_0 = tracker.update(
        boxes=[np.array([10, 10, 50, 50], dtype=np.float32)],
        embeddings=[embedding],
        labels=["red helmet"],
        frame_index=0,
        scores=[0.9],
    )
    ids_1 = tracker.update(
        boxes=[np.array([22, 12, 62, 52], dtype=np.float32)],
        embeddings=[unit([0.99, 0.11, 0.01])],
        labels=["person wearing a red helmet"],
        frame_index=1,
        scores=[0.88],
    )

    assert ids_0 == [1]
    assert ids_1 == [1]


def test_botsort_creates_new_identity_for_unrelated_object():
    tracker = BotSort(max_age=2, use_cmc=False)
    first = tracker.update(
        boxes=[np.array([0, 0, 20, 20], dtype=np.float32)],
        embeddings=[unit([1.0, 0.0, 0.0])],
        labels=["drone"],
        frame_index=0,
        scores=[0.9],
    )
    second = tracker.update(
        boxes=[np.array([200, 200, 230, 230], dtype=np.float32)],
        embeddings=[unit([0.0, 1.0, 0.0])],
        labels=["drone"],
        frame_index=1,
        scores=[0.9],
    )

    assert first == [1]
    assert second == [2]

"""BoT-SORT tracker with CLIP ReID embeddings.

Implements the BoT-SORT association recipe:
- Kalman motion prediction
- Camera-motion compensation (sparse optical flow)
- High/low confidence dual-stage matching (ByteTrack-style)
- IoU + appearance (ReID) fusion for the first stage
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box_a in enumerate(a):
        for j, box_b in enumerate(b):
            out[i, j] = box_iou(box_a, box_b)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, 1.0 - cosine_similarity(a, b))


def _xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w / h, h], dtype=np.float64)


def _xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah.astype(float)
    h = max(1e-3, h)
    w = max(1e-3, a * h)
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)


def _apply_affine_to_xyxy(box: np.ndarray, warp: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float64,
    )
    transformed = corners @ warp.T
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height)."""

    _ndim = 4
    _dt = 1.0

    def __init__(self, box: np.ndarray) -> None:
        ndim = self._ndim
        self._motion = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion[i, ndim + i] = self._dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self._std_weight_pos = 1.0 / 20.0
        self._std_weight_vel = 1.0 / 160.0

        mean = np.zeros(2 * ndim, dtype=np.float64)
        mean[:ndim] = _xyxy_to_xyah(box)
        std = [
            2.0 * self._std_weight_pos * mean[3],
            2.0 * self._std_weight_pos * mean[3],
            1e-2,
            2.0 * self._std_weight_pos * mean[3],
            10.0 * self._std_weight_vel * mean[3],
            10.0 * self._std_weight_vel * mean[3],
            1e-5,
            10.0 * self._std_weight_vel * mean[3],
        ]
        self.mean = mean
        self.covariance = np.diag(np.square(std))

    def predict(self) -> np.ndarray:
        std_pos = [
            self._std_weight_pos * self.mean[3],
            self._std_weight_pos * self.mean[3],
            1e-2,
            self._std_weight_pos * self.mean[3],
        ]
        std_vel = [
            self._std_weight_vel * self.mean[3],
            self._std_weight_vel * self.mean[3],
            1e-5,
            self._std_weight_vel * self.mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        self.mean = self._motion @ self.mean
        self.covariance = self._motion @ self.covariance @ self._motion.T + motion_cov
        return self.to_xyxy()

    def update(self, box: np.ndarray) -> None:
        measurement = _xyxy_to_xyah(box)
        std = [
            self._std_weight_pos * self.mean[3],
            self._std_weight_pos * self.mean[3],
            1e-1,
            self._std_weight_pos * self.mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        projected_cov = self._update_mat @ self.covariance @ self._update_mat.T + innovation_cov
        kalman_gain = np.linalg.solve(
            projected_cov.T,
            (self.covariance @ self._update_mat.T).T,
        ).T
        innovation = measurement - self._update_mat @ self.mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ projected_cov @ kalman_gain.T

    def apply_camera_motion(self, warp: np.ndarray) -> None:
        box = self.to_xyxy()
        warped = _apply_affine_to_xyxy(box, warp)
        self.mean[:4] = _xyxy_to_xyah(warped)
        self.mean[4:6] = warp[:2, :2] @ self.mean[4:6]

    def to_xyxy(self) -> np.ndarray:
        return _xyah_to_xyxy(self.mean[:4])


class CameraMotionCompensator:
    """Sparse-optical-flow affine CMC used by BoT-SORT."""

    def __init__(self, max_points: int = 500) -> None:
        self.max_points = max_points
        self._prev_gray: np.ndarray | None = None

    def estimate(self, frame_bgr: np.ndarray | None) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float64)
        if frame_bgr is None:
            return identity
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return identity

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) < 8:
            self._prev_gray = gray
            return identity

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if next_pts is None or status is None:
            self._prev_gray = gray
            return identity

        ok = status.reshape(-1) == 1
        if int(ok.sum()) < 8:
            self._prev_gray = gray
            return identity

        src = prev_pts.reshape(-1, 2)[ok]
        dst = next_pts.reshape(-1, 2)[ok]
        warp, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        self._prev_gray = gray
        return identity if warp is None else warp.astype(np.float64)


@dataclass
class STrack:
    track_id: int
    kalman: KalmanBoxTracker
    embedding: np.ndarray
    label: str
    score: float
    last_seen: int
    state: str = "tracked"
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    embedding_gallery: list[np.ndarray] = field(default_factory=list)

    @property
    def bbox(self) -> np.ndarray:
        return self.kalman.to_xyxy()

    def predict(self) -> None:
        self.kalman.predict()
        self.age += 1
        if self.state != "tracked":
            self.time_since_update += 1

    def apply_cmc(self, warp: np.ndarray) -> None:
        self.kalman.apply_camera_motion(warp)

    def update(
        self,
        bbox: np.ndarray,
        embedding: np.ndarray,
        label: str,
        score: float,
        frame_index: int,
        embedding_momentum: float = 0.9,
        gallery_size: int = 50,
    ) -> None:
        self.kalman.update(bbox)
        mixed = embedding_momentum * self.embedding + (1.0 - embedding_momentum) * embedding
        norm = float(np.linalg.norm(mixed))
        self.embedding = (mixed / norm).astype(np.float32) if norm > 0 else mixed.astype(np.float32)
        self.embedding_gallery.append(self.embedding.copy())
        if len(self.embedding_gallery) > gallery_size:
            self.embedding_gallery = self.embedding_gallery[-gallery_size:]
        self.label = label
        self.score = float(score)
        self.last_seen = frame_index
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"

    def mark_lost(self) -> None:
        self.state = "lost"

    def reid_distance(self, embedding: np.ndarray) -> float:
        gallery = self.embedding_gallery or [self.embedding]
        return float(min(cosine_distance(feat, embedding) for feat in gallery))


def _linear_assignment(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    unmatched_a = set(range(cost.shape[0]))
    unmatched_b = set(range(cost.shape[1]))
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if float(cost[row, col]) > thresh:
            continue
        matches.append((row, col))
        unmatched_a.discard(row)
        unmatched_b.discard(col)
    return matches, sorted(unmatched_a), sorted(unmatched_b)


class BotSort:
    """BoT-SORT using CLIP crop embeddings as ReID features."""

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.05,
        new_track_thresh: float = 0.25,
        match_thresh: float = 0.80,
        second_match_thresh: float = 0.50,
        proximity_thresh: float = 0.50,
        appearance_thresh: float = 0.40,
        max_age: int = 30,
        use_cmc: bool = True,
        fuse_iou: bool = True,
        embedding_momentum: float = 0.9,
        gallery_size: int = 50,
        max_cost: float | None = None,
        iou_weight: float = 0.45,
        embedding_weight: float = 0.55,
        label_weight: float = 0.0,
        min_iou_gate: float = 0.01,
        min_embedding_gate: float = 0.0,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh if max_cost is None else float(max_cost)
        self.second_match_thresh = second_match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.max_age = max_age
        self.use_cmc = use_cmc
        self.fuse_iou = fuse_iou
        self.embedding_momentum = embedding_momentum
        self.gallery_size = gallery_size
        self.iou_weight = iou_weight
        self.embedding_weight = embedding_weight
        self.label_weight = label_weight
        self.min_iou_gate = min_iou_gate
        self.min_embedding_gate = min_embedding_gate
        self._tracks: dict[int, STrack] = {}
        self._next_id = 1
        self._cmc = CameraMotionCompensator()

    @property
    def tracks(self) -> dict[int, STrack]:
        return self._tracks

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = embedding.astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm > 0 else embedding

    def _active_tracks(self) -> list[STrack]:
        return [t for t in self._tracks.values() if t.state in {"tracked", "lost"}]

    def _fused_cost(
        self,
        tracks: list[STrack],
        boxes: list[np.ndarray],
        embeddings: list[np.ndarray],
        labels: list[str],
        use_reid: bool,
    ) -> np.ndarray:
        if not tracks or not boxes:
            return np.zeros((len(tracks), len(boxes)), dtype=np.float32)

        track_boxes = np.stack([t.bbox for t in tracks], axis=0)
        det_boxes = np.stack(boxes, axis=0)
        ious = pairwise_iou(track_boxes, det_boxes)
        cost = np.full((len(tracks), len(boxes)), 1e6, dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, (embedding, label) in enumerate(zip(embeddings, labels)):
                iou = float(ious[i, j])
                iou_dist = 1.0 - iou

                if not use_reid:
                    if iou < self.min_iou_gate:
                        continue
                    cost[i, j] = iou_dist
                    continue

                reid = track.reid_distance(embedding)
                # BoT-SORT: appearance only helps when geometry is close enough.
                if iou_dist > self.proximity_thresh and reid > self.appearance_thresh:
                    continue
                if reid > self.appearance_thresh and iou < self.min_iou_gate:
                    continue
                if self.min_embedding_gate > 0 and (1.0 - reid) < self.min_embedding_gate and iou < self.min_iou_gate:
                    continue

                if self.fuse_iou:
                    fused = self.embedding_weight * reid + self.iou_weight * iou_dist
                else:
                    fused = reid if reid <= self.appearance_thresh else iou_dist
                if self.label_weight > 0 and track.label.casefold() != label.casefold():
                    fused += self.label_weight
                cost[i, j] = fused
        return cost

    def update(
        self,
        boxes: list[np.ndarray],
        embeddings: list[np.ndarray],
        labels: list[str],
        frame_index: int,
        scores: list[float] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> list[int]:
        if not (len(boxes) == len(embeddings) == len(labels)):
            raise ValueError("boxes, embeddings, and labels must have equal lengths")
        if scores is None:
            scores = [1.0] * len(boxes)
        if len(scores) != len(boxes):
            raise ValueError("scores must match detections")

        embeddings = [self._normalize(emb) for emb in embeddings]
        scores_f = [float(s) for s in scores]

        active = self._active_tracks()
        for track in active:
            track.predict()

        warp = self._cmc.estimate(frame_bgr) if self.use_cmc else np.eye(2, 3, dtype=np.float64)
        if self.use_cmc:
            for track in active:
                track.apply_cmc(warp)

        if not boxes:
            for track in active:
                if track.time_since_update > 0:
                    track.mark_lost()
            self._remove_stale(frame_index)
            return []

        high_idx = [i for i, s in enumerate(scores_f) if s >= self.track_high_thresh]
        low_idx = [
            i
            for i, s in enumerate(scores_f)
            if self.track_low_thresh <= s < self.track_high_thresh
        ]

        matches_h, unmatched_tracks, unmatched_high = self._associate(
            active,
            [boxes[i] for i in high_idx],
            [embeddings[i] for i in high_idx],
            [labels[i] for i in high_idx],
            thresh=self.match_thresh,
            use_reid=True,
        )

        assigned: dict[int, int] = {}
        for t_i, d_local in matches_h:
            det_i = high_idx[d_local]
            track = active[t_i]
            track.update(
                boxes[det_i],
                embeddings[det_i],
                labels[det_i],
                scores_f[det_i],
                frame_index,
                embedding_momentum=self.embedding_momentum,
                gallery_size=self.gallery_size,
            )
            assigned[det_i] = track.track_id

        remain_tracks = [active[i] for i in unmatched_tracks]
        matches_l, unmatched_tracks2, _ = self._associate(
            remain_tracks,
            [boxes[i] for i in low_idx],
            [embeddings[i] for i in low_idx],
            [labels[i] for i in low_idx],
            thresh=self.second_match_thresh,
            use_reid=False,
        )
        for local_t, d_local in matches_l:
            det_i = low_idx[d_local]
            track = remain_tracks[local_t]
            track.update(
                boxes[det_i],
                embeddings[det_i],
                labels[det_i],
                scores_f[det_i],
                frame_index,
                embedding_momentum=self.embedding_momentum,
                gallery_size=self.gallery_size,
            )
            assigned[det_i] = track.track_id

        for track in (remain_tracks[i] for i in unmatched_tracks2):
            track.mark_lost()

        for d_local in unmatched_high:
            det_i = high_idx[d_local]
            if scores_f[det_i] < self.new_track_thresh:
                continue
            assigned[det_i] = self._create_track(
                boxes[det_i],
                embeddings[det_i],
                labels[det_i],
                scores_f[det_i],
                frame_index,
            )

        self._remove_stale(frame_index)

        output_ids: list[int] = []
        for det_i in range(len(boxes)):
            if det_i in assigned:
                output_ids.append(assigned[det_i])
            else:
                output_ids.append(
                    self._create_track(
                        boxes[det_i],
                        embeddings[det_i],
                        labels[det_i],
                        scores_f[det_i],
                        frame_index,
                    )
                )
        return output_ids

    def _associate(
        self,
        tracks: list[STrack],
        boxes: list[np.ndarray],
        embeddings: list[np.ndarray],
        labels: list[str],
        thresh: float,
        use_reid: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        cost = self._fused_cost(tracks, boxes, embeddings, labels, use_reid=use_reid)
        return _linear_assignment(cost, thresh=thresh)

    def _create_track(
        self,
        box: np.ndarray,
        embedding: np.ndarray,
        label: str,
        score: float,
        frame_index: int,
    ) -> int:
        track_id = self._next_id
        self._next_id += 1
        embedding = self._normalize(embedding)
        self._tracks[track_id] = STrack(
            track_id=track_id,
            kalman=KalmanBoxTracker(box),
            embedding=embedding,
            label=label,
            score=float(score),
            last_seen=frame_index,
            embedding_gallery=[embedding.copy()],
        )
        return track_id

    def _remove_stale(self, frame_index: int) -> None:
        stale = [
            tid
            for tid, track in self._tracks.items()
            if track.time_since_update > self.max_age or frame_index - track.last_seen > self.max_age
        ]
        for tid in stale:
            del self._tracks[tid]


# Backward-compatible alias for existing tests / imports.
SemanticTracker = BotSort

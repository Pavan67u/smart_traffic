"""
Simple IoU-based tracker fallback. Maintains track ids by greedy IoU matching.
This is intended as a CPU-friendly fallback when ByteTrack is not available.
"""
import numpy as np
from typing import List, Dict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_thresh = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, cls, lost}

    def update(self, dets: List[List[float]]):
        # dets: list of [x1,y1,x2,y2,score,cls]
        assigned = {}
        det_bboxes = [d[:4] for d in dets]
        det_cls = [int(d[5]) if len(d) > 5 else None for d in dets]

        if not self.tracks:
            for i, bbox in enumerate(det_bboxes):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': bbox, 'cls': det_cls[i], 'lost': 0}
            return [{'track_id': tid, 'bbox': v['bbox'], 'cls': v['cls']} for tid, v in self.tracks.items()]

        # build IoU matrix
        track_ids = list(self.tracks.keys())
        iou_mat = np.zeros((len(track_ids), len(det_bboxes)), dtype=float)
        for ti, tid in enumerate(track_ids):
            for di, db in enumerate(det_bboxes):
                iou_mat[ti, di] = iou(self.tracks[tid]['bbox'], db)

        # greedy matching
        matches = []
        used_det = set()
        for ti, tid in enumerate(track_ids):
            best_di = int(iou_mat[ti].argmax()) if det_bboxes else -1
            if best_di >= 0 and iou_mat[ti, best_di] >= self.iou_thresh and best_di not in used_det:
                matches.append((tid, best_di))
                used_det.add(best_di)

        # update matched
        for tid, di in matches:
            self.tracks[tid]['bbox'] = det_bboxes[di]
            self.tracks[tid]['cls'] = det_cls[di]
            self.tracks[tid]['lost'] = 0

        # unmatched dets -> new tracks
        for di in range(len(det_bboxes)):
            if di not in used_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': det_bboxes[di], 'cls': det_cls[di], 'lost': 0}

        # age lost tracks
        to_del = []
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] >= self.max_lost:
                to_del.append(tid)
            else:
                self.tracks[tid]['lost'] += 1

        for tid in to_del:
            del self.tracks[tid]

        return [{'track_id': tid, 'bbox': v['bbox'], 'cls': v['cls']} for tid, v in self.tracks.items()]

def create_fallback_tracker():
    return SimpleIoUTracker()

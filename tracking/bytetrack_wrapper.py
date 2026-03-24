"""
Minimal ByteTrack wrapper placeholder. This file provides a thin wrapper API
so other code can import update_tracker(dets, frame) and receive track dicts.

Note: an external ByteTrack implementation must be installed and imported here
for production. This wrapper tries to import common ByteTrack packages and
falls back to raising an informative error.
"""
from typing import List, Dict
try:
    # try common bytetrack package locations (user may have installed different packages)
    from bytetrack import BYTETracker
except Exception:
    BYTETracker = None

class ByteTrackWrapper:
    def __init__(self, fps: int = 30):
        if BYTETracker is None:
            raise RuntimeError('ByteTrack package not installed. Install bytetrack-py or provide wrapper.')
        self.tracker = BYTETracker()

    def update(self, dets: List[List[float]], frame) -> List[Dict]:
        """
        dets: list of [x1,y1,x2,y2,score,cls]
        frame: numpy array or None (some trackers require frame size)
        returns: list of tracks: {track_id, bbox, cls, score}
        """
        tracks = self.tracker.update(dets, frame.shape[:2])
        out = []
        for tr in tracks:
            # Adapt depending on tracker return format
            out.append({
                'track_id': int(tr[0]),
                'bbox': [float(tr[1]), float(tr[2]), float(tr[3]), float(tr[4])],
                'score': float(tr[5]) if len(tr) > 5 else None,
                'cls': int(tr[6]) if len(tr) > 6 else None,
            })
        return out

def create_tracker(fps: int = 30):
    return ByteTrackWrapper(fps=fps)

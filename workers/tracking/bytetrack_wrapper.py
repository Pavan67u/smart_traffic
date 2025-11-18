# Placeholder wrapper; integrate an existing ByteTrack implementation.
# Exposes update(detections) -> list of Track(id, bbox, path, t)
from collections import deque


class Track:
	def __init__(self, tid, bbox, t):
		self.id = tid
		self.bbox = bbox
		self.path = deque(maxlen=32)
		self.t = t


class ByteTrackWrapper:
	def __init__(self):
		pass

	def update(self, detections, t):
		# TODO: plug real bytetrack; for now, 1:1 mock IDs
		tracks = []
		for i, det in enumerate(detections):
			x1,y1,x2,y2,_cls,conf = det
			tr = Track(i, (x1,y1,x2,y2), t)
			cx, cy = (x1+x2)/2, (y1+y2)/2
			tr.path.append((cx, cy))
			tracks.append(tr)
		return tracks
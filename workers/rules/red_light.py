import shapely.geometry as geom
import shapely.ops as ops
import numpy as np


class RedLightRule:
	def __init__(self, cfg):
		a, b = cfg['stop_line']
		self.stop_line = geom.LineString([tuple(a), tuple(b)])
		self.red_zone = geom.Polygon([tuple(p) for p in cfg['red_zone_polygon']])
		self.state = 'G'
		self.state_history = []

	def update_light(self, state):
		self.state_history.append(state)
		self.state_history = self.state_history[-5:] # debounce window
		self.state = max(set(self.state_history), key=self.state_history.count)

	def evaluate(self, tracks):
		events = []
		for trk in tracks:
			if len(trk.path) < 2:
				continue
			p1 = geom.Point(trk.path[-2])
			p2 = geom.Point(trk.path[-1])
			seg = geom.LineString([p1, p2])
			crossed = seg.crosses(self.stop_line) or seg.touches(self.stop_line)
			now_inside = self.red_zone.contains(p2)
			was_inside = self.red_zone.contains(p1)
			if crossed and now_inside and not was_inside and self.state in ('R','Y'):
				events.append({
					'type':'red_light',
					'track_id': trk.id,
					'time': trk.t,
					'bbox': trk.bbox,
					'state': self.state
				})
		return events
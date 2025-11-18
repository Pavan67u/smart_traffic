import cv2
import json
import os
import hashlib
import time


class EvidenceBuilder:
	def __init__(self, out_dir="evidence"):
		self.out_dir = out_dir
		os.makedirs(self.out_dir, exist_ok=True)

	def _hash(self, p):
		h = hashlib.sha256()
		with open(p, 'rb') as f:
			h.update(f.read())
		return h.hexdigest()

	def save_packet(self, frame, crop, state, meta):
		ts = int(time.time() * 1000)
		base = os.path.join(self.out_dir, f"case_{ts}")
		os.makedirs(base, exist_ok=True)
		p1 = os.path.join(base, 'overview.jpg')
		p2 = os.path.join(base, 'crop.jpg')
		cv2.imwrite(p1, frame)
		cv2.imwrite(p2, crop)
		packet = {
			'overview': p1,
			'crop': p2,
			'hashes': {
				'overview': self._hash(p1),
				'crop': self._hash(p2)
			},
			'light_state': state,
			**meta
		}
		with open(os.path.join(base, 'evidence.json'), 'w') as f:
			json.dump(packet, f, indent=2)
		return base
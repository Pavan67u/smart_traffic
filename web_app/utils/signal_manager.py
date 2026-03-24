import time
import threading
from datetime import datetime

class SignalManager:
    """
    Manages the state of the traffic signal (RED, GREEN, YELLOW).
    Thread-safe implementation for API + Background Timer.
    """
    def __init__(self, mode='manual', red_duration=10, green_duration=10, yellow_duration=3):
        self.mode = mode  # 'manual' or 'timer'
        self.state = 'RED'
        self.last_changed = time.time()
        
        # Configuration for timer mode
        self.durations = {
            'RED': red_duration,
            'GREEN': green_duration,
            'YELLOW': yellow_duration
        }
        self.next_state_map = {
            'RED': 'GREEN',
            'GREEN': 'YELLOW',
            'YELLOW': 'RED'
        }
        
        self.lock = threading.Lock()
        
        # Start background timer thread
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.thread.start()

    def _timer_loop(self):
        while not self.stop_event.is_set():
            if self.mode == 'timer':
                with self.lock:
                    elapsed = time.time() - self.last_changed
                    if elapsed >= self.durations.get(self.state, 5):
                        self._transition(self.next_state_map.get(self.state, 'RED'))
            time.sleep(0.1)

    def _transition(self, new_state):
        # Internal method, assumes lock is held if called from locked context
        self.state = new_state
        self.last_changed = time.time()
        print(f"[SignalManager] State changed to {self.state}")

    def set_state(self, new_state):
        with self.lock:
            if new_state.upper() in ['RED', 'GREEN', 'YELLOW']:
                self.mode = 'manual' # Force manual on specific set
                self._transition(new_state.upper())
                return True
            return False

    def set_mode(self, mode):
        with self.lock:
            if mode in ['manual', 'timer']:
                self.mode = mode
                return True
            return False

    def get_status(self):
        with self.lock:
            return {
                'state': self.state,
                'mode': self.mode,
                'last_changed': datetime.fromtimestamp(self.last_changed).isoformat(),
                'elapsed': round(time.time() - self.last_changed, 2)
            }
            
# Global instance
SIGNAL_MANAGER = SignalManager()

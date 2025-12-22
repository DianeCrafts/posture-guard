import time


class TemporalSmoother:
    def __init__(self, confirm_time=1.5):
        self.confirm_time = confirm_time

        self.stable_state = None
        self.state_start_time = None

        self.candidate_state = None
        self.candidate_start_time = None

    def update(self, classification):
        """
        Input: classification dict from PostureClassifier
        Output: dict with stable state + duration
        """
        if not classification:
            return None

        current_state = classification["state"]
        now = time.time()

        # First frame initialization
        if self.stable_state is None:
            self.stable_state = current_state
            self.state_start_time = now
            return self._result(now)

        # If state matches stable state, reset candidate
        if current_state == self.stable_state:
            self.candidate_state = None
            self.candidate_start_time = None
            return self._result(now)

        # New candidate state
        if self.candidate_state != current_state:
            self.candidate_state = current_state
            self.candidate_start_time = now
            return self._result(now)

        # Candidate state persists long enough → commit
        if now - self.candidate_start_time >= self.confirm_time:
            self.stable_state = self.candidate_state
            self.state_start_time = now
            self.candidate_state = None
            self.candidate_start_time = None

        return self._result(now)

    def _result(self, now):
        duration = now - self.state_start_time
        return {
            "stable_state": self.stable_state,
            "duration_sec": duration,
        }

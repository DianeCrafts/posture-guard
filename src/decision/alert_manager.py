import time


class AlertManager:
    def __init__(self, bad_posture_time=5.0, cooldown_time=15.0):
        self.bad_posture_time = bad_posture_time
        self.cooldown_time = cooldown_time

        self.last_alert_time = None

    def check(self, smoothed_state):
        """
        Returns True if alert should trigger.
        """
        if not smoothed_state:
            return False

        state = smoothed_state["stable_state"]
        duration = smoothed_state["duration_sec"]
        now = time.time()

        if state != "BAD":
            return False

        if duration < self.bad_posture_time:
            return False

        if self.last_alert_time is not None:
            if now - self.last_alert_time < self.cooldown_time:
                return False

        self.last_alert_time = now
        return True

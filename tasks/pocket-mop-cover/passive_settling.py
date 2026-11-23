"""Observe an uninterrupted passive hold without changing any physical state."""

import math


class PassiveHold:
    """Require native-step evidence before ending at a 20 Hz control boundary."""

    def __init__(self, physics_hz):
        assert physics_hz >= 20 and physics_hz % 20 == 0
        self.hz = physics_hz
        self.steps = 0
        self.contiguous_steps = 0
        self.first_stop_step = None

    def observe(self, native_speed, difference_speed, inclination, wing_angle, height):
        values = (native_speed, difference_speed, inclination, wing_angle, height)
        assert all(math.isfinite(value) for value in values), 'Nonfinite passive measurement'
        assert native_speed >= 0 and difference_speed >= 0
        valid = (native_speed <= .03 and difference_speed <= .03 and
                 inclination <= 3. and wing_angle <= 3. and -.001 <= height <= .01)
        self.contiguous_steps = self.contiguous_steps + 1 if valid else 0
        self.steps += 1
        stop = self.contiguous_steps >= 2*self.hz and self.steps % (self.hz//20) == 0
        if stop and self.first_stop_step is None:
            self.first_stop_step = self.steps
        return stop

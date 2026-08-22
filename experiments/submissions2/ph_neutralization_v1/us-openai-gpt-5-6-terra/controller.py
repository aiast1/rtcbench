import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)

        self.u_start = 14.22

        # The analyzer delay is roughly 20 s, so this remains a conservative
        # delayed-process PI controller, but is faster than the prior tuning.
        self.kp = 0.60
        self.ki = 0.0080
        self.kaw = 0.040

        self.du_up = 0.42
        self.du_down = 0.65

        self.reset()

    def reset(self):
        self.initialized = False
        self.u_prev = self.u_start
        self.trim = 0.0

        self.last_ph = 6.5
        self.last_level = 22.0
        self.last_r = 6.5
        self.level_rate = 0.0
        self.last_t = None

    def _feedforward(self, ph_sp):
        p = float(np.clip(ph_sp, 4.0, 10.5))
        p0 = 6.5

        pK1 = 6.35
        pK2 = 10.25

        a1 = 1.0 / (1.0 + 10.0 ** (pK1 - p))
        a2 = 1.0 / (1.0 + 10.0 ** (pK2 - p))
        a10 = 1.0 / (1.0 + 10.0 ** (pK1 - p0))
        a20 = 1.0 / (1.0 + 10.0 ** (pK2 - p0))

        return 14.22 + 5.45 * ((a1 - a10) + (a2 - a20))

    def _level_cap(self, level, level_rate):
        h = float(level)
        cap = 19.4

        if h > 25.7:
            cap -= 1.55 * (h - 25.7)

        if level_rate > 0.0:
            cap -= 10.0 * level_rate

        if h >= 28.3:
            cap = min(cap, 14.0)
        if h >= 28.9:
            cap = min(cap, 9.5)
        if h >= 29.4:
            cap = min(cap, 4.5)

        return float(np.clip(cap, 0.0, 19.4))

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if r.size > 0 and np.isfinite(r[0]):
            self.last_r = float(r[0])
        ph_sp = float(np.clip(self.last_r, 4.0, 10.5))

        ph_good = (
            y.size >= 1
            and quality.size >= 1
            and bool(quality[0])
            and np.isfinite(y[0])
        )
        level_good = (
            y.size >= 2
            and quality.size >= 2
            and bool(quality[1])
            and np.isfinite(y[1])
        )

        if ph_good:
            self.last_ph = float(np.clip(y[0], 0.0, 14.0))

        if level_good:
            level = float(np.clip(y[1], 0.0, 35.0))
        else:
            level = self.last_level

        if not self.initialized:
            self.initialized = True
            self.last_level = level
            self.last_t = float(t)
            return np.array([self.u_start], dtype=float)

        dt = self.dt
        if self.last_t is not None:
            observed_dt = float(t) - self.last_t
            if np.isfinite(observed_dt) and observed_dt > 0.1:
                dt = float(np.clip(observed_dt, 0.5 * self.dt, 2.0 * self.dt))

        if level_good:
            raw_rate = (level - self.last_level) / max(dt, 1.0e-6)
            self.level_rate = 0.75 * self.level_rate + 0.25 * raw_rate
            self.last_level = level
        else:
            self.level_rate *= 0.90

        ph = self.last_ph
        error = ph_sp - ph

        raw_u = self._feedforward(ph_sp) + self.trim + self.kp * error

        upper = min(30.0, self._level_cap(level, self.level_rate))
        lower = 0.0

        # Retain a margin to the true discharge limit while avoiding an
        # unnecessary governor action around the 9.9 pH operating target.
        if ph_good and ph > 10.10:
            upper = min(upper, self.u_prev - 0.68 * (ph - 10.10))
        if ph_good and ph > 10.30:
            upper = min(upper, self.u_prev - 0.75)

        if ph_good and ph < 4.45 and level < 28.3:
            lower = min(18.0, self.u_prev)

        target = float(np.clip(raw_u, lower, upper))

        u_cmd = float(np.clip(
            target,
            self.u_prev - self.du_down,
            self.u_prev + self.du_up,
        ))
        u_cmd = float(np.clip(u_cmd, 0.0, 30.0))

        if ph_good:
            self.trim += dt * (self.ki * error + self.kaw * (u_cmd - raw_u))
            self.trim = float(np.clip(self.trim, -7.0, 7.0))

        self.u_prev = u_cmd
        self.last_t = float(t)

        return np.array([u_cmd], dtype=float)
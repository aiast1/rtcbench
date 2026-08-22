import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nom = float(brief.sample_time)
        self.u_start = np.array([14.19, -1113.5], dtype=float)
        self.reset()

    def reset(self):
        self.u = self.u_start.copy()
        self.last_t = None
        self.started = False
        self.cb_f = np.nan
        self.t_f = np.nan
        self.cb_bias = 0.0
        self.t_bias = 0.0

    @staticmethod
    def _good(x, q):
        return bool(q) and np.isfinite(x)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if self.last_t is None:
            dt = self.dt_nom
        else:
            dt = float(np.clip(float(t) - self.last_t, 1.0, 30.0))
        self.last_t = float(t)

        cb_ok = len(y) > 0 and len(quality) > 0 and self._good(y[0], quality[0])
        tt_ok = len(y) > 1 and len(quality) > 1 and self._good(y[1], quality[1])

        if cb_ok:
            if not np.isfinite(self.cb_f):
                self.cb_f = float(y[0])
            else:
                self.cb_f += 0.31 * (float(y[0]) - self.cb_f)

        if tt_ok:
            if not np.isfinite(self.t_f):
                self.t_f = float(y[1])
            else:
                self.t_f += 0.30 * (float(y[1]) - self.t_f)

        r_cb = float(r[0]) if len(r) > 0 and np.isfinite(r[0]) else 1.09
        r_t = float(r[1]) if len(r) > 1 and np.isfinite(r[1]) else 114.19

        if not self.started:
            self.started = True
            return self.u.copy()

        # High-flow branch feedforward.  The explicit schedule feedforward is
        # important because the B analyzer has transport delay.
        drop = max(0.0, 1.090 - r_cb)
        f_ff = 14.70 + np.sqrt(drop / 0.00058)
        f_ff = float(np.clip(f_ff, 14.10, 34.70))

        if cb_ok and np.isfinite(self.cb_f):
            e_cb = self.cb_f - r_cb
            e_int = 0.0 if abs(e_cb) < 0.003 else e_cb

            # This trim is intentionally appreciably faster than a nominal
            # model correction: kinetic draws cause meaningful steady errors
            # at the low-product, high-flow operating points.
            bias_try = self.cb_bias + 0.020 * e_int * dt
            f_try = f_ff + 11.5 * e_cb + bias_try

            if not ((f_try >= 34.70 and e_cb > 0.0) or
                    (f_try <= 14.10 and e_cb < 0.0)):
                self.cb_bias = float(np.clip(bias_try, -5.0, 10.0))

            f_target = f_ff + 11.5 * e_cb + self.cb_bias
        else:
            f_target = f_ff + self.cb_bias

        f_target = float(np.clip(f_target, 14.10, 34.70))

        # Higher dilution brings in cold feed; this feedforward prevents the
        # temperature loop from unnecessarily fighting scheduled F/V moves.
        q_ff = -1113.5 + 115.0 * (f_target - 14.19)

        if tt_ok and np.isfinite(self.t_f):
            e_t = self.t_f - r_t
            e_db = 0.0 if abs(e_t) < 0.045 else e_t

            bias_try = self.t_bias - 1.45 * e_db * dt
            q_try = q_ff - 315.0 * e_db + bias_try

            if not ((q_try <= -8900.0 and e_db > 0.0) or
                    (q_try >= -20.0 and e_db < 0.0)):
                self.t_bias = float(np.clip(bias_try, -7200.0, 1800.0))

            q_target = q_ff - 315.0 * e_db + self.t_bias

            # Reactor-temperature protection also provides margin for the
            # unmeasured jacket temperature constraint.
            if self.t_f > 133.0:
                q_target = min(q_target, -2600.0 - 420.0 * (self.t_f - 133.0))
            if self.t_f > 142.0:
                q_target = min(q_target, -6200.0 - 500.0 * (self.t_f - 142.0))
            if self.t_f > 147.0:
                q_target = -9000.0
        else:
            q_target = q_ff + self.t_bias

        q_target = float(np.clip(q_target, -9000.0, 0.0))

        # Fast enough for analyzer-delayed concentration transitions, while
        # retaining a bounded smooth command suitable for sticky valves.
        max_df = 0.65 * dt / self.dt_nom
        max_dq = 110.0 * dt / self.dt_nom

        df = float(np.clip(f_target - self.u[0], -max_df, max_df))
        dq = float(np.clip(q_target - self.u[1], -max_dq, max_dq))

        if np.isfinite(self.t_f) and self.t_f > 147.5:
            dq = -9000.0 - self.u[1]

        self.u[0] = float(np.clip(self.u[0] + df, 3.0, 35.0))
        self.u[1] = float(np.clip(self.u[1] + dq, -9000.0, 0.0))
        return self.u.copy()
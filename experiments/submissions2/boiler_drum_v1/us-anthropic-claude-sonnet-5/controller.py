import numpy as np


class Controller:
    """
    Three-element style feedwater control for a natural-circulation drum
    boiler with shrink/swell behaviour.

    Structure:
      - Inner loop: valve position is PI-controlled to track a FEED FLOW
        setpoint measured via FT-104.  This closes the loop on actual mass
        flow, so it is largely insensitive to the valve's pressure-dependent
        gain (the "mismatch" between commanded % and delivered kg/s).
      - Outer loop: a slow PI trim on indicated level (LT-101) adjusts the
        feed-flow setpoint around the current steam flow (FT-103).  Because
        the trim is small and slow, most of the transient feedwater demand
        during load changes is met by direct steam-flow feedforward
        (mass-balance), which is exactly what prevents the shrink/swell
        illusion from being chased by an aggressive level loop.
      - Soft safety backoff terms bias the flow setpoint down as indicated
        level approaches the high trip and up as the (crude) inventory
        estimate approaches the low trip, bounded so they cannot combine
        into a large control kick.
      - Output is rate-limited and lightly low-pass filtered to avoid
        chatter / duty violations while remaining responsive.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0))
        self.u_min = 0.0
        self.u_max = 100.0
        self.u_start = 45.0

        # measurement filters (s)
        self.tau_level = 25.0
        self.tau_flow = 12.0

        # outer level trim PI -> produces a feed-flow bias (kg/s)
        self.Kp_lvl = 0.035        # kg/s per mm
        self.Ki_lvl = 0.00045      # kg/s per mm per s
        self.trim_limit = 12.0     # kg/s clamp on trim

        # soft safety backoff (kg/s), smooth ramps starting inside the trip
        self.ceiling_start = 140.0   # mm, indicated level
        self.ceiling_full = 220.0
        self.ceiling_max_bias = 25.0  # kg/s max cut

        self.floor_start = -140.0    # mm, inventory estimate
        self.floor_full = -220.0
        self.floor_max_bias = 25.0   # kg/s max add

        # inner flow PI -> produces valve % command
        self.Kp_flow = 1.1        # % per (kg/s)
        self.Ki_flow = 0.10       # % per (kg/s) per s

        # output shaping
        self.max_rate = 1.0        # %/s hard rate limit
        self.out_lp_alpha = 0.35

        # crude inventory estimator time constant for slow re-anchoring
        self.inv_conv = 0.05       # mm per (kg/s * s) mismatch accumulation
        self.inv_anchor_tau = 400.0

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.level_f = None
        self.steam_f = None
        self.fw_f = None

        self.trim_int = 0.0
        self.flow_int = self.u_start

        self.inv_est = 0.0

        self.u_prev = self.u_start
        self.u_lp = self.u_start

    # ------------------------------------------------------------------
    @staticmethod
    def _lp_update(prev, new_val, alpha, good):
        if prev is None:
            return float(new_val)
        if not good:
            return prev
        return prev + alpha * (float(new_val) - prev)

    @staticmethod
    def _smooth_ramp(x, start, full, max_out):
        # 0 below `start` (in the safe direction), ramps to max_out at `full`
        if full == start:
            return 0.0
        if full > start:
            frac = (x - start) / (full - start)
        else:
            frac = (start - x) / (start - full)
        frac = float(np.clip(frac, 0.0, 1.0))
        return max_out * frac

    # ------------------------------------------------------------------
    def step(self, t, y, r, quality):
        dt = self.dt

        level_raw = float(y[0])
        steam_raw = float(y[2]) if len(y) > 2 else np.nan
        fw_raw = float(y[3]) if len(y) > 3 else np.nan

        q_level = bool(quality[0]) if len(quality) > 0 else True
        q_steam = bool(quality[2]) if len(quality) > 2 else True
        q_fw = bool(quality[3]) if len(quality) > 3 else True

        alpha_level = dt / (self.tau_level + dt)
        alpha_flow = dt / (self.tau_flow + dt)

        self.level_f = self._lp_update(self.level_f, level_raw, alpha_level, q_level)
        self.steam_f = self._lp_update(self.steam_f, steam_raw, alpha_flow, q_steam)
        self.fw_f = self._lp_update(self.fw_f, fw_raw, alpha_flow, q_fw)

        level_f = self.level_f if self.level_f is not None else level_raw
        steam_f = self.steam_f if self.steam_f is not None else 50.0
        fw_f = self.fw_f if self.fw_f is not None else self.u_start

        # ---- crude inventory estimate (mass-balance drift + slow anchor) ----
        if q_steam and q_fw:
            self.inv_est += self.inv_conv * (fw_f - steam_f) * dt
        anchor_alpha = dt / (self.inv_anchor_tau + dt)
        self.inv_est += anchor_alpha * (level_f - self.inv_est)

        # ---- setpoint ----
        sp = r[0] if (r is not None and len(r) > 0 and not np.isnan(r[0])) else 0.0

        # ---- outer level trim PI (slow, small authority) ----
        err_level = sp - level_f
        trim_p = self.Kp_lvl * err_level
        trim_i_candidate = self.trim_int + self.Ki_lvl * err_level * dt
        trim_i_candidate = float(np.clip(trim_i_candidate, -self.trim_limit, self.trim_limit))

        trim_unclamped = trim_p + trim_i_candidate
        trim = float(np.clip(trim_unclamped, -self.trim_limit, self.trim_limit))

        # anti-windup for the trim integrator
        if not ((trim_unclamped > self.trim_limit and err_level > 0) or
                (trim_unclamped < -self.trim_limit and err_level < 0)):
            self.trim_int = trim_i_candidate

        # ---- soft safety backoff (bounded, cannot stack unboundedly) ----
        ceiling_bias = -self._smooth_ramp(level_f, self.ceiling_start,
                                           self.ceiling_full, self.ceiling_max_bias)
        floor_bias = self._smooth_ramp(self.inv_est, self.floor_start,
                                        self.floor_full, self.floor_max_bias)

        safety_bias = float(np.clip(ceiling_bias + floor_bias, -self.ceiling_max_bias,
                                     self.floor_max_bias))

        # ---- feed-flow setpoint: mass-balance feedforward + trim + safety ----
        flow_sp = steam_f + trim + safety_bias
        flow_sp = float(np.clip(flow_sp, 0.0, 100.0))

        # ---- inner flow PI -> valve command ----
        err_flow = flow_sp - fw_f
        u_unclamped = self.flow_int + self.Kp_flow * err_flow
        u_clamped = float(np.clip(u_unclamped, self.u_min, self.u_max))

        sat_high = u_unclamped > self.u_max
        sat_low = u_unclamped < self.u_min
        if not ((sat_high and err_flow > 0) or (sat_low and err_flow < 0)):
            self.flow_int += self.Ki_flow * err_flow * dt
            self.flow_int = float(np.clip(self.flow_int, self.u_min, self.u_max))

        u_desired = u_clamped

        # ---- output shaping: rate limit + low-pass to avoid chatter ----
        delta_desired = u_desired - self.u_prev
        max_step = self.max_rate * dt
        delta_applied = float(np.clip(delta_desired, -max_step, max_step))
        u_target = self.u_prev + delta_applied

        self.u_lp = self.u_lp + self.out_lp_alpha * (u_target - self.u_lp)
        u_final = float(np.clip(self.u_lp, self.u_min, self.u_max))

        self.u_prev = u_final

        return np.array([u_final], dtype=float)
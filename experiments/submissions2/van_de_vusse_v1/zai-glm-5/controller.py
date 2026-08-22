import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Actuator limits
        self.u_min = np.array([3.0, -9000.0])
        self.u_max = np.array([35.0, 0.0])
        
        # Initial actuator position (bumpless start)
        self.u = np.array([14.19, -1113.5])
        
        # Safety limits
        self.T_max = 150.0  # Reactor temperature limit
        self.TK_max = 150.0  # Jacket temperature limit (unmeasured)
        
        # PID gains for C_B (channel 0) - F/V manipulation
        # C_B has non-monotonic response, need careful tuning
        self.Kp_cb = 0.8
        self.Ki_cb = 0.03
        self.Kd_cb = 0.05
        
        # PID gains for T (channel 1) - Q_K manipulation
        self.Kp_t = 80.0
        self.Ki_t = 8.0
        self.Kd_t = 5.0
        
        # Integral states
        self.integral_cb = 0.0
        self.integral_t = 0.0
        
        # Previous errors for derivative
        self.prev_error_cb = 0.0
        self.prev_error_t = 0.0
        
        # Anti-windup back-calculation coefficients
        self.Kb_cb = 0.5
        self.Kb_t = 0.5
        
        # Filter coefficient for derivative
        self.N_deriv = 10.0
        
        # Derivative filter states
        self.deriv_filter_cb = 0.0
        self.deriv_filter_t = 0.0
        
        # Reference filter states (for smooth setpoint tracking)
        self.ref_filter_cb = 1.09
        self.ref_filter_t = 114.19
        self.ref_filter_tau = 3.0  # Time constant for reference filter
        
        # Output rate limits (for smooth actuation)
        self.rate_limit_fv = 2.0  # h^-1 per sample
        self.rate_limit_qk = 500.0  # kJ/h per sample
        
        # Previous valid measurements
        self.prev_y_cb = None
        self.prev_y_t = None
        
        # Tracking for derivative on measurement (not error)
        self.prev_y_cb_deriv = None
        self.prev_y_t_deriv = None
        
        # Jacket temperature estimation parameters
        self.TK_estimate = 114.19  # Initial estimate
        self.TK_filter_coeff = 0.1
        
        # Ramp detection
        self.ramp_active = False
        self.ramp_start_time = 2900.0
        self.ramp_end_time = 3500.0
        
        # Quality tracking
        self.bad_quality_count = 0
        self.max_bad_samples = 5
        
        # Previous output for anti-windup
        self.prev_u_fv = 14.19
        self.prev_u_qk = -1113.5
        
    def reset(self):
        """Reset controller state for new scenario."""
        self.u = np.array([14.19, -1113.5])
        self.integral_cb = 0.0
        self.integral_t = 0.0
        self.prev_error_cb = 0.0
        self.prev_error_t = 0.0
        self.deriv_filter_cb = 0.0
        self.deriv_filter_t = 0.0
        self.ref_filter_cb = 1.09
        self.ref_filter_t = 114.19
        self.prev_y_cb = None
        self.prev_y_t = None
        self.prev_y_cb_deriv = None
        self.prev_y_t_deriv = None
        self.TK_estimate = 114.19
        self.bad_quality_count = 0
        self.prev_u_fv = 14.19
        self.prev_u_qk = -1113.5
        
    def _clamp(self, val, low, high):
        """Clamp value to range."""
        return np.clip(val, low, high)
    
    def _estimate_jacket_temp(self, T, Q_K):
        """Estimate jacket temperature from energy balance.
        TK ≈ T - Q_K / (kw * AR) for steady-state approximation.
        """
        # Nominal parameters
        kw = 4032.0
        AR = 0.215
        UA = kw * AR
        
        if abs(UA) > 1e-6:
            TK_est = T - Q_K / UA
        else:
            TK_est = T
            
        # Filter the estimate
        self.TK_estimate = (1 - self.TK_filter_coeff) * self.TK_estimate + self.TK_filter_coeff * TK_est
        return self.TK_estimate
    
    def _pid_compute(self, error, integral, prev_error, Kp, Ki, Kd, 
                     deriv_filter, Ts, use_deriv_on_meas=False, measurement=None, prev_measurement=None):
        """Compute PID output with derivative filtering."""
        # Proportional
        P = Kp * error
        
        # Integral (will be updated externally with anti-windup)
        I = Ki * integral
        
        # Derivative (on measurement to avoid setpoint kick)
        if use_deriv_on_meas and measurement is not None and prev_measurement is not None:
            deriv_input = -(measurement - prev_measurement) / Ts
        else:
            deriv_input = (error - prev_error) / Ts
            
        # Filtered derivative
        alpha = self.N_deriv * Ts / (1 + self.N_deriv * Ts)
        deriv_filter_new = alpha * deriv_input + (1 - alpha) * deriv_filter
        D = Kd * deriv_filter_new
        
        return P + I + D, deriv_filter_new
    
    def step(self, t, y, r, quality):
        """Execute one control step."""
        Ts = self.sample_time
        
        # Extract measurements
        y_cb = y[0] if quality[0] else self.prev_y_cb if self.prev_y_cb is not None else 1.09
        y_t = y[1] if quality[1] else self.prev_y_t if self.prev_y_t is not None else 114.19
        
        # Update previous measurements
        if quality[0]:
            self.prev_y_cb = y_cb
        if quality[1]:
            self.prev_y_t = y_t
            
        # Extract setpoints (handle NaN)
        r_cb = r[0] if not np.isnan(r[0]) else 1.09
        r_t = r[1] if not np.isnan(r[1]) else 114.19
        
        # Reference filtering for smooth tracking
        ref_filter_alpha = Ts / (self.ref_filter_tau + Ts)
        self.ref_filter_cb = (1 - ref_filter_alpha) * self.ref_filter_cb + ref_filter_alpha * r_cb
        self.ref_filter_t = (1 - ref_filter_alpha) * self.ref_filter_t + ref_filter_alpha * r_t
        
        # Use filtered reference for error calculation
        ref_cb = self.ref_filter_cb
        ref_t = self.ref_filter_t
        
        # Compute errors
        error_cb = ref_cb - y_cb
        error_t = ref_t - y_t
        
        # === C_B Controller (F/V manipulation) ===
        # Non-monotonic response: need to stay on correct side of maximum
        # The maximum is around F/V = 14.7, we operate near 14.19
        
        # Store previous measurements for derivative
        prev_y_cb_deriv = self.prev_y_cb_deriv if self.prev_y_cb_deriv is not None else y_cb
        self.prev_y_cb_deriv = y_cb
        
        # Compute PID for C_B
        u_cb_raw, self.deriv_filter_cb = self._pid_compute(
            error_cb, self.integral_cb, self.prev_error_cb,
            self.Kp_cb, self.Ki_cb, self.Kd_cb,
            self.deriv_filter_cb, Ts,
            use_deriv_on_meas=True, measurement=y_cb, prev_measurement=prev_y_cb_deriv
        )
        
        # Base F/V from initial operating point
        u_fv_base = 14.19
        u_fv = u_fv_base + u_cb_raw
        
        # === Temperature Controller (Q_K manipulation) ===
        prev_y_t_deriv = self.prev_y_t_deriv if self.prev_y_t_deriv is not None else y_t
        self.prev_y_t_deriv = y_t
        
        # Compute PID for T
        u_t_raw, self.deriv_filter_t = self._pid_compute(
            error_t, self.integral_t, self.prev_error_t,
            self.Kp_t, self.Ki_t, self.Kd_t,
            self.deriv_filter_t, Ts,
            use_deriv_on_meas=True, measurement=y_t, prev_measurement=prev_y_t_deriv
        )
        
        # Base Q_K from initial operating point
        u_qk_base = -1113.5
        u_qk = u_qk_base + u_t_raw
        
        # === Safety constraints ===
        # Estimate jacket temperature
        TK_est = self._estimate_jacket_temp(y_t, u_qk)
        
        # Temperature safety: reduce Q_K if T or TK approaching limit
        T_safety_margin = 3.0  # degC margin
        TK_safety_margin = 5.0  # degC margin for unmeasured variable
        
        # If reactor temperature approaching limit, increase cooling
        if y_t > self.T_max - T_safety_margin:
            # Need more cooling
            cooling_needed = (y_t - (self.T_max - T_safety_margin)) * 200
            u_qk = min(u_qk, -cooling_needed)
            
        # If jacket temperature estimate approaching limit, reduce cooling intensity
        if TK_est > self.TK_max - TK_safety_margin:
            # Limit how cold the jacket can be
            max_cooling = -(y_t - (self.TK_max - TK_safety_margin)) * 500
            u_qk = max(u_qk, max_cooling)
            
        # === Apply rate limits ===
        u_fv = self._clamp(u_fv, self.prev_u_fv - self.rate_limit_fv, self.prev_u_fv + self.rate_limit_fv)
        u_qk = self._clamp(u_qk, self.prev_u_qk - self.rate_limit_qk, self.prev_u_qk + self.rate_limit_qk)
        
        # === Apply saturation limits ===
        u_fv_sat = self._clamp(u_fv, self.u_min[0], self.u_max[0])
        u_qk_sat = self._clamp(u_qk, self.u_min[1], self.u_max[1])
        
        # === Anti-windup: back-calculation ===
        # Update integrals with anti-windup
        saturation_error_cb = u_fv_sat - u_fv
        saturation_error_t = u_qk_sat - u_qk
        
        # Update integrals
        self.integral_cb += error_cb * Ts + self.Kb_cb * saturation_error_cb * Ts
        self.integral_t += error_t * Ts + self.Kb_t * saturation_error_t * Ts
        
        # Limit integrals to prevent excessive windup
        max_integral_cb = 5.0 / max(self.Ki_cb, 1e-6)
        max_integral_t = 500.0 / max(self.Ki_t, 1e-6)
        self.integral_cb = self._clamp(self.integral_cb, -max_integral_cb, max_integral_cb)
        self.integral_t = self._clamp(self.integral_t, -max_integral_t, max_integral_t)
        
        # Update previous errors
        self.prev_error_cb = error_cb
        self.prev_error_t = error_t
        
        # Store outputs
        self.prev_u_fv = u_fv_sat
        self.prev_u_qk = u_qk_sat
        
        # Build output
        self.u = np.array([u_fv_sat, u_qk_sat])
        
        return self.u
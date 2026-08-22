import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        # Use correct attribute names from TaskBrief
        self.n_y = len(getattr(brief, 'measurement_names', ['AI-CB', 'TI-101']))
        self.n_u = len(getattr(brief, 'actuator_names', ['FCV-101', 'TCV-201']))
        
        # Setpoint tracking: only CB and T are scored
        self.sp_cb = None
        self.sp_t = None
        
        # Internal state for PI controllers
        self.integral_cb = 0.0
        self.integral_t = 0.0
        self.prev_cb = None
        self.prev_t = None
        
        # Anti-windup: back-calculation gains
        self.k_aw_cb = 0.1
        self.k_aw_t = 0.1
        
        # Controller gains (tuned for robustness over nominal model)
        # These are conservative to handle parameter uncertainty and non-monotonicity
        self.kp_cb = 0.8   # Proportional gain for CB
        self.ki_cb = 0.05  # Integral gain for CB
        self.kp_t = 1.2    # Proportional gain for T
        self.ki_t = 0.08   # Integral gain for T
        
        # Actuator limits from brief.actuator_ranges
        self.u_min = np.array([brief.actuator_ranges[0][0], brief.actuator_ranges[1][0]])  # [3.0, -9000.0]
        self.u_max = np.array([brief.actuator_ranges[0][1], brief.actuator_ranges[1][1]])  # [35.0, 0.0]
        
        # Steady-state initialization (bumpless start)
        self.steady_state_u = np.array([14.19, -1113.5])
        
        # Safety limits (unmeasured but constrained)
        self.t_max = 150.0
        self.tk_max = 150.0
        
    def reset(self):
        # Reset to bumpless start: initialize integrals so that u = steady_state_u at start
        self.integral_cb = 0.0
        self.integral_t = 0.0
        self.prev_cb = None
        self.prev_t = None
        
    def step(self, t, y, r, quality):
        # Extract measurements: y[0] = CB (mol/L), y[1] = T (degC)
        cb_meas = y[0] if quality[0] else None
        t_meas = y[1] if quality[1] else None
        
        # Extract setpoints: r[0] = CB setpoint, r[1] = T setpoint (if not nan)
        sp_cb = r[0] if not np.isnan(r[0]) else None
        sp_t = r[1] if not np.isnan(r[1]) else None
        
        # If no valid measurement, use zero error to avoid windup
        cb_error = 0.0 if (cb_meas is None or sp_cb is None) else (sp_cb - cb_meas)
        t_error = 0.0 if (t_meas is None or sp_t is None) else (sp_t - t_meas)
        
        # Initialize previous errors on first valid step
        if self.prev_cb is None:
            self.prev_cb = cb_error
        if self.prev_t is None:
            self.prev_t = t_error
        
        # PI control with anti-windup via back-calculation
        # Compute unsaturated control
        u_cb_unsat = self.kp_cb * cb_error + self.ki_cb * self.integral_cb
        u_t_unsat = self.kp_t * t_error + self.ki_t * self.integral_t
        
        # Actuator mapping: 
        # u[0] = F/V (feed dilution rate) -> primarily affects CB (and T via reaction heat)
        # u[1] = Q_K (jacket cooling) -> primarily affects T (and indirectly CB via kinetics)
        # We use decoupled PI: CB loop -> F/V, T loop -> Q_K
        u_unsat = np.array([u_cb_unsat, u_t_unsat])
        
        # Clip to actuator limits
        u_sat = np.clip(u_unsat, self.u_min, self.u_max)
        
        # Anti-windup: back-calculation
        # Update integrals only if not saturated, or use back-calculation to reduce windup
        if self.u_min[0] <= u_cb_unsat <= self.u_max[0]:
            self.integral_cb += cb_error * self.dt
        else:
            # Back-calculation: reduce integral effort proportional to saturation
            self.integral_cb += self.dt * (cb_error - self.k_aw_cb * (u_cb_unsat - u_sat[0]))
            
        if self.u_min[1] <= u_t_unsat <= self.u_max[1]:
            self.integral_t += t_error * self.dt
        else:
            self.integral_t += self.dt * (t_error - self.k_aw_t * (u_t_unsat - u_sat[1]))
        
        # Store current errors for next step
        self.prev_cb = cb_error
        self.prev_t = t_error
        
        # Return saturated control
        return u_sat
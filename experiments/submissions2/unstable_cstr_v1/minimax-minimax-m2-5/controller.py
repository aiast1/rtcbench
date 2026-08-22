import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = brief.sample_time  # 3.0 s
        
        # Conservative PID gains for unstable CSTR
        # Lower gains to avoid pushing into cold state
        self.Kp = 0.8
        self.Ki = 0.02
        self.Kd = 2.0
        
        # Anti-windup parameters
        self.Kt = 0.5
        
        # Actuator limits
        self.u_min = 270.0
        self.u_max = 340.0
        self.slew_max = 1.5 * self.Ts  # 1.5 K/s -> 4.5 K per step
        
        # State
        self.integrator = 0.0
        self.u_prev = 300.0
        self.prev_t = None
        self.prev_tr = None
        self.dtr_filtered = 0.0
        self.filter_alpha = 0.2  # Heavy low-pass filter for derivative
        
    def reset(self):
        """Called before each scenario for bumpless start"""
        self.integrator = 0.0
        self.u_prev = 300.0
        self.prev_t = None
        self.prev_tr = None
        self.dtr_filtered = 0.0
        
    def step(self, t, y, r, quality):
        # y[0] = TI-101 reactor temperature (scored)
        # y[1] = TI-102 jacket supply temperature (not scored)
        
        # Handle measurement quality - use previous value if stale
        tr = y[0]
        if not quality[0]:
            tr = self.prev_tr if self.prev_tr is not None else 350.0
        
        # Setpoint from r[0] (TI-101 is scored)
        sp = r[0] if not np.isnan(r[0]) else 350.0
        
        # Initialize on first call
        if self.prev_t is None:
            self.prev_t = t
            self.prev_tr = tr
            return np.array([self.u_prev])
        
        # Calculate error
        error = sp - tr
        
        # Derivative on measurement with heavy low-pass filter
        dtr_raw = (tr - self.prev_tr) / self.Ts if self.prev_tr is not None else 0.0
        self.dtr_filtered = self.filter_alpha * dtr_raw + (1 - self.filter_alpha) * self.dtr_filtered
        
        # PID control law
        u = self.Kp * error + self.Ki * self.integrator - self.Kd * self.dtr_filtered
        
        # Saturation
        u_sat = np.clip(u, self.u_min, self.u_max)
        
        # Back-calculation anti-windup
        if u != u_sat and self.Ki > 0:
            self.integrator += (error - (u_sat - u) * self.Kt / self.Kp) * self.Ts
        else:
            self.integrator += error * self.Ts
        
        # Clamp integrator to prevent excessive windup
        self.integrator = np.clip(self.integrator, -30.0, 30.0)
        
        u = u_sat
        
        # Apply slew rate limit
        delta_u = u - self.u_prev
        if abs(delta_u) > self.slew_max:
            u = self.u_prev + np.sign(delta_u) * self.slew_max
        
        # Final clip after slew limit
        u = np.clip(u, self.u_min, self.u_max)
        
        # Update state
        self.prev_t = t
        self.prev_tr = tr
        self.u_prev = u
        
        return np.array([u])
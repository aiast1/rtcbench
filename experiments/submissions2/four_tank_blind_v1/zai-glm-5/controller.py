import numpy as np

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # PI gains - tuned for four-tank dynamics
        self.Kp = np.array([1.5, 1.5])
        self.Ki = np.array([0.08, 0.08])
        
        # State
        self.integral = np.zeros(2)
        self.u_prev = np.array([3.0, 3.0])
        self.y_prev = None
        self.initialized = False
        
        # Safety margins for unmeasured tanks
        self.h_safe_upper = 18.5
        self.h_safe_lower = 1.5
        
        # Filter for measurement noise
        self.y_filtered = None
        self.filter_alpha = 0.4
        
    def reset(self):
        self.integral = np.zeros(2)
        self.u_prev = np.array([3.0, 3.0])
        self.y_prev = None
        self.initialized = False
        self.y_filtered = None
        
    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        quality = np.asarray(quality, dtype=bool)
        
        # Handle bad quality readings
        if self.y_prev is not None:
            for i in range(len(y)):
                if not quality[i] or np.isnan(y[i]):
                    y[i] = self.y_prev[i]
        
        # Clamp measurements
        y = np.clip(y, 0.0, 20.0)
        
        # Handle NaN setpoints
        r_valid = np.where(np.isnan(r), y, r)
        r_valid = np.clip(r_valid, 0.0, 20.0)
        
        # Initialize on first valid step
        if not self.initialized:
            if np.all(quality) and not np.any(np.isnan(y)):
                self.y_prev = y.copy()
                self.y_filtered = y.copy()
                # Bumpless transfer - initialize integral to match current output
                e_init = r_valid - y
                self.integral = (self.u_prev - self.Kp * e_init) / (self.Ki + 1e-9)
                self.integral = np.clip(self.integral, -6.0, 6.0)
                self.initialized = True
            return self.u_prev.copy()
        
        # Filter measurements
        if self.y_filtered is not None:
            self.y_filtered = self.filter_alpha * y + (1.0 - self.filter_alpha) * self.y_filtered
        else:
            self.y_filtered = y.copy()
        
        # Use filtered measurements for control
        y_ctrl = self.y_filtered.copy()
        
        # Compute error
        e = r_valid - y_ctrl
        
        # Proportional term
        P = self.Kp * e
        
        # Integral term with anti-windup
        I = self.Ki * self.integral
        
        # Calculate unsaturated output
        u_unsat = P + I
        
        # Apply saturation
        u_sat = np.clip(u_unsat, self.u_min, self.u_max)
        
        # Anti-windup: conditional integration
        for i in range(2):
            if quality[i]:
                # Check saturation
                if u_sat[i] >= self.u_max[i] - 0.01:
                    # At upper limit - only integrate negative error
                    if e[i] < 0:
                        self.integral[i] += e[i] * self.Ts
                elif u_sat[i] <= self.u_min[i] + 0.01:
                    # At lower limit - only integrate positive error
                    if e[i] > 0:
                        self.integral[i] += e[i] * self.Ts
                else:
                    # Not saturated - normal integration
                    self.integral[i] += e[i] * self.Ts
        
        # Clamp integral
        self.integral = np.clip(self.integral, -6.0, 6.0)
        
        # Recalculate output with clamped integral
        I = self.Ki * self.integral
        u = self.Kp * e + I
        
        # Safety constraints for unmeasured tanks
        for i in range(2):
            if y_ctrl[i] > self.h_safe_upper:
                # Reduce output to prevent overflow
                factor = 1.0 - 0.4 * (y_ctrl[i] - self.h_safe_upper) / (20.0 - self.h_safe_upper)
                u[i] = min(u[i], self.u_prev[i] * factor + 0.05)
            
            if y_ctrl[i] < self.h_safe_lower:
                # Increase output to prevent emptying
                factor = 1.0 + 0.6 * (self.h_safe_lower - y_ctrl[i]) / self.h_safe_lower
                u[i] = max(u[i], self.u_prev[i] * factor)
        
        # Rate limiting for smooth actuation
        max_rate = 1.0  # V per control period
        du = u - self.u_prev
        du = np.clip(du, -max_rate, max_rate)
        u = self.u_prev + du
        
        # Final saturation
        u = np.clip(u, self.u_min, self.u_max)
        
        # Store for next iteration
        self.u_prev = u.copy()
        self.y_prev = y.copy()
        
        return u
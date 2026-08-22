import numpy as np

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time
        self.umax = np.array([10.0, 10.0])
        self.umin = np.array([0.0, 0.0])
        self.u_init = np.array([3.0, 3.0])
        
        # Conservative PI tuning - robust to unknown plant dynamics
        # Low gain to avoid aggressive action on noisy/delayed measurements
        self.Kp = np.array([0.3, 0.3])
        self.Ki = np.array([0.02, 0.02])
        
        # Anti-windup: limit integral term
        self.int_max = np.array([2.0, 2.0])
        
        # Rate limiting to prevent actuator duty limit violation (0.017)
        # At 2s sample time, max change per step = 0.017 * 2 = 0.034
        self.du_max = np.array([0.03, 0.03])
        
        # Measurement filter (simple first-order)
        self.alpha = 0.3
        
        self.reset()
        
    def reset(self):
        self.u = self.u_init.copy()
        self.u_prev = self.u.copy()
        self.integral = np.array([0.0, 0.0])
        self.y_filt = np.array([12.26, 12.78])  # Initial guess near steady state
        
    def step(self, t, y, r, quality):
        # Handle bad quality measurements: use filtered value
        y_eff = np.where(quality, y, self.y_filt)
        
        # Low-pass filter to reduce noise impact
        self.y_filt = self.alpha * y_eff + (1 - self.alpha) * self.y_filt
        
        # Compute error (handle NaN in setpoints)
        e = np.where(np.isnan(r), 0.0, r - self.y_filt)
        
        # Update integral with anti-windup
        self.integral += e * self.Ki * self.Ts
        self.integral = np.clip(self.integral, -self.int_max, self.int_max)
        
        # PI control law
        u_pi = self.u_init + self.Kp * e + self.integral
        
        # Rate limiting
        du = u_pi - self.u_prev
        du_clipped = np.clip(du, -self.du_max, self.du_max)
        u_rate_limited = self.u_prev + du_clipped
        
        # Apply actuator limits
        u_final = np.clip(u_rate_limited, self.umin, self.umax)
        
        # Anti-windup: back-calculation
        # If saturation occurred, limit integral growth
        u_sat = u_final - u_rate_limited
        if np.any(u_sat != 0):
            self.integral -= u_sat * 0.5
        
        self.u_prev = self.u.copy()
        self.u = u_final
        
        return self.u
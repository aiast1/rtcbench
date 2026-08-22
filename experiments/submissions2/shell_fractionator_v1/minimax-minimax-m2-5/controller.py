import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.ts = brief.sample_time
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Further increased gains for better tracking
        self.Kp = np.array([0.35, 0.28, 0.40])
        self.Ki = np.array([0.009, 0.007, 0.011])
        self.Kd = np.array([2.0, 1.5, 2.5])
        
        # Derivative filter
        self.filter_alpha = 0.20
        
        # Anti-windup
        self.integrator_max = np.array([0.5, 0.5, 0.5])
        
        # Actuator duty limit
        self.max_du = 0.016
        
        # State variables
        self.reset()
        
    def reset(self):
        """Initialize states before each scenario"""
        self.u = np.array([0.0, 0.0, 0.0])
        self.integrator = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([np.nan, np.nan, np.nan])
        self.prev_filtered_d = np.array([0.0, 0.0, 0.0])
        self.y_good = None
        
    def step(self, t, y, r, quality):
        """
        Main control loop
        t: current time
        y: measured outputs (3,)
        r: setpoints (3,), nan for non-scored
        quality: bool array (3,), False = stale/bad
        """
        # Handle bad quality measurements
        y_filtered = y.copy()
        for i in range(3):
            if not quality[i]:
                if self.y_good is not None:
                    y_filtered[i] = self.y_good[i]
                else:
                    y_filtered[i] = 0.0
        
        self.y_good = y_filtered.copy()
        
        # Compute errors
        error = np.zeros(3)
        for i in range(3):
            if not np.isnan(r[i]):
                error[i] = r[i] - y_filtered[i]
            else:
                error[i] = 0.0
        
        # Initialize on first call
        if np.any(np.isnan(self.prev_error)):
            self.prev_error = error.copy()
            return self.u.copy()
        
        # Compute filtered derivative
        de = (error - self.prev_error) / self.ts
        
        filtered_d = np.zeros(3)
        for i in range(3):
            filtered_d[i] = self.filter_alpha * de[i] + (1 - self.filter_alpha) * self.prev_filtered_d[i]
        
        self.prev_error = error.copy()
        self.prev_filtered_d = filtered_d.copy()
        
        # PID terms
        P = self.Kp * error
        
        # Integral with anti-windup
        for i in range(3):
            sat_pos = self.u[i] >= self.u_max[i] - 0.001
            sat_neg = self.u[i] <= self.u_min[i] + 0.001
            
            if (error[i] > 0 and sat_pos) or (error[i] < 0 and sat_neg):
                pass
            else:
                self.integrator[i] += error[i] * self.ts
        
        # Clamp integrator
        self.integrator = np.clip(self.integrator, -self.integrator_max, self.integrator_max)
        
        I = self.Ki * self.integrator
        D = self.Kd * filtered_d
        
        # Compute control
        u_new = P + I + D
        
        # Apply limits
        u_new = np.clip(u_new, self.u_min, self.u_max)
        
        # Rate limiting
        du = u_new - self.u
        du_magnitude = np.sum(np.abs(du))
        
        if du_magnitude > self.max_du:
            scale = self.max_du / du_magnitude
            u_new = self.u + du * scale
        
        u_new = np.clip(u_new, self.u_min, self.u_max)
        
        self.u = u_new
        
        return self.u.copy()
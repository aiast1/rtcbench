import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # More aggressive tuning for better tracking
        self.Kp = np.array([3.5, 3.5])
        self.Ki = np.array([0.04, 0.04])
        self.Kd = np.array([0.0, 0.0])
        
        # Integral windup limits
        self.int_min = np.array([-1.0, -1.0])
        self.int_max = np.array([1.0, 1.0])
        
        # Actuator limits
        self.u_min = np.array([1.5, 2.0])
        self.u_max = np.array([4.5, 5.0])
        
        # Nominal operating point
        self.u_nominal = np.array([2.70629, 3.20629])
        
        # Safety margins for unmeasured constraints
        self.safety_margin_D = 0.10
        self.safety_margin_B = 0.10
        
        # Filter parameters
        self.setpoint_filter = 0.25
        self.y_filter_alpha = 0.3
        
        # Feed rate
        self.F = 1.0
        
        # Rate limit
        self.max_rate = 0.0028
        
    def reset(self):
        self.integral = np.zeros(2)
        self.y_prev = None
        self.r_filtered_prev = np.array([0.99, 0.99])
        self.error_prev = np.zeros(2)
        self.u_prev = self.u_nominal.copy()
        
    def step(self, t, y, r, quality):
        if self.y_prev is None:
            self.y_prev = y.copy()
        
        # Filter measurements
        y_filt = self.y_filter_alpha * y + (1 - self.y_filter_alpha) * self.y_prev
        self.y_prev = y_filt.copy()
        y_use = np.where(quality, y_filt, self.y_prev)
        
        # Filter setpoints
        r_clean = np.where(np.isnan(r), self.r_filtered_prev, r)
        r_filt = self.setpoint_filter * r_clean + (1 - self.setpoint_filter) * self.r_filtered_prev
        self.r_filtered_prev = r_filt.copy()
        
        # Compute errors
        error = r_filt - y_use
        
        # Update integrator
        self.integral += self.Ki * error * self.dt
        
        # PID output
        u_delta = self.Kp * error + self.integral
        
        # Nominal plus delta
        u = self.u_nominal + u_delta
        
        # Apply safety constraints
        L, V = u[0], u[1]
        
        for _ in range(10):
            D = V - L
            B = L + self.F - V
            
            # Check hard constraints first (0.05 minimum)
            if D < 0.06:
                deficit = 0.07 - D
                V += deficit * 0.6
                L -= deficit * 0.4
            elif D < self.safety_margin_D:
                deficit = self.safety_margin_D - D
                V += deficit * 0.5
                L -= deficit * 0.5
            
            if B < 0.06:
                deficit = 0.07 - B
                L += deficit * 0.6
                V -= deficit * 0.4
            elif B < self.safety_margin_B:
                deficit = self.safety_margin_B - B
                L += deficit * 0.5
                V -= deficit * 0.5
            
            # Clip to actuator limits
            L = np.clip(L, self.u_min[0], self.u_max[0])
            V = np.clip(V, self.u_min[1], self.u_max[1])
        
        u = np.array([L, V])
        
        # Anti-windup
        u_clipped = np.clip(u, self.u_min, self.u_max)
        if np.any(u_clipped != u):
            saturation = u_clipped - u
            self.integral -= 0.4 * saturation
        
        self.integral = np.clip(self.integral, self.int_min, self.int_max)
        
        u_out = np.clip(u, self.u_min, self.u_max)
        
        # Rate limit
        delta_u = u_out - self.u_prev
        delta_u_clipped = np.clip(delta_u, -self.max_rate * self.dt, self.max_rate * self.dt)
        u_out = self.u_prev + delta_u_clipped
        
        self.u_prev = u_out.copy()
        
        return u_out
import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.L0 = brief.actuators_start[0]
        self.V0 = brief.actuators_start[1]
        self.Kp = 0.25  # Reduced gain for robustness
        self.Ti = 300.0  # Longer reset time
        self.Ts = brief.sample_time
        self.reset()

    def reset(self):
        self.integral_L = self.L0
        self.integral_V = self.V0
        self.last_y = np.array([0.99, 0.99])
        self.last_r = np.array([0.99, 0.99])
        self.last_u = np.array([self.L0, self.V0])

    def step(self, t, y, r, quality):
        # Handle bad measurements and setpoint changes
        y_clean = np.where(quality, y, self.last_y)
        r_clean = np.where(~np.isnan(r), r, self.last_r)
        
        # Store last good values
        self.last_y = y_clean
        self.last_r = r_clean
        
        # Calculate errors
        e = r_clean - y_clean
        
        # PI control with conditional integration
        if quality[0]:
            self.integral_L += (self.Kp/self.Ti) * e[0] * self.Ts
        if quality[1]:
            self.integral_V += (self.Kp/self.Ti) * e[1] * self.Ts
        
        # Calculate preliminary outputs
        u_L = self.Kp * e[0] + self.integral_L
        u_V = self.Kp * e[1] + self.integral_V
        
        # Apply D/B flow constraints (V-L ≥ 0.05 and L+1-V ≥ 0.05)
        u_L_clipped = np.clip(u_L, 1.5, 4.5)
        min_V = u_L_clipped + 0.05
        max_V = u_L_clipped + 0.95
        u_V_clipped = np.clip(u_V, np.maximum(min_V, 2.0), np.minimum(max_V, 5.0))
        
        # Anti-windup back-calculation
        self.integral_L += (u_L_clipped - u_L)/self.Kp
        self.integral_V += (u_V_clipped - u_V)/self.Kp
        
        # Rate limiting (max 0.003 kmol/min per second)
        du = np.array([u_L_clipped, u_V_clipped]) - self.last_u
        du_limited = np.clip(du, -0.003*self.Ts, 0.003*self.Ts)
        u_final = self.last_u + du_limited
        
        # Final actuator limits
        u_final[0] = np.clip(u_final[0], 1.5, 4.5)
        final_min_V = u_final[0] + 0.05
        final_max_V = u_final[0] + 0.95
        u_final[1] = np.clip(u_final[1], np.maximum(final_min_V, 2.0), np.minimum(final_max_V, 5.0))
        
        self.last_u = u_final
        return u_final
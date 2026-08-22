import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.Kp = [0.3, 0.3]  # Conservative P gains
        self.Ki = [0.004, 0.004]  # Reduced integral action
        self.u_min = 0.0
        self.u_max = 7.0  # Lower max output for safety margin
        self.prev_y = np.array([11.28, 11.94])
        self.integral = np.zeros(2)
        self.rate_limit = 0.3  # Stricter rate limiting
        self.last_u = np.array([3.0, 3.0])
        self.safe_zone = 18.0  # Start limiting before overflow
        self.emergency_reduce = 15.0  # Aggressively cut input at high levels

    def reset(self):
        self.integral = np.clip(self.last_u, self.u_min, self.u_max)
        self.prev_y = np.array([11.28, 11.94])
        self.last_u = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Handle bad measurements with persistence
        current_y = np.where(quality, y, self.prev_y)
        self.prev_y = current_y
        
        # Error calculation with setpoint filtering
        e = np.nan_to_num(r - current_y, nan=0.0)
        
        # Anti-windup: conditional integration
        for i in range(2):
            if (self.last_u[i] < self.u_max) and (self.last_u[i] > self.u_min):
                self.integral[i] += self.Ki[i] * e[i] * self.sample_time
            self.integral[i] = np.clip(self.integral[i], self.u_min*0.5, self.u_max*0.5)

        # Conservative control calculation
        u_calc = self.Kp * e + self.integral
        
        # Dynamic safety constraints
        safety_multiplier = np.ones(2)
        for i in range(2):
            if current_y[i] > self.emergency_reduce:
                safety_multiplier[i] = 0.5
            elif current_y[i] > self.safe_zone:
                safety_multiplier[i] = 0.7 + 0.3*(20.0 - current_y[i])/(20.0 - self.safe_zone)
        
        # Apply rate limiting and final clamping
        u_target = np.clip(u_calc * safety_multiplier, self.u_min, self.u_max)
        delta_u = np.clip(u_target - self.last_u, -self.rate_limit, self.rate_limit)
        u = np.clip(self.last_u + delta_u, self.u_min, self.u_max)
        self.last_u = u
        
        return u
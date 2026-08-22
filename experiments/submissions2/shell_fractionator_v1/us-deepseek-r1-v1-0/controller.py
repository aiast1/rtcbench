import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.model = brief.model_hint
        self.Kp = np.zeros(3)
        self.Ki = np.zeros(3)
        self.integral = np.zeros(3)
        self.last_good_y = np.zeros(3)
        self.prev_u = np.zeros(3)
        self.actuator_limits = (-0.5, 0.5)
        self.sample_time = brief.sample_time
        
        # Access model parameters using attribute access
        for i in range(3):
            K = self.model.K[i][i]
            tau = self.model.TAU[i][i]
            L = self.model.L[i][i]
            lambda_ = max(L, 0.1*tau)
            self.Kp[i] = (0.9*tau)/(K*(lambda_ + L))
            self.Ki[i] = self.Kp[i]/(0.8*tau)

    def reset(self):
        self.integral = np.zeros(3)
        self.last_good_y = np.zeros(3)
        self.prev_u = np.zeros(3)

    def step(self, t, y, r, quality):
        y = np.array(y)
        # Handle stale measurements
        for i in range(3):
            if quality[i]:
                self.last_good_y[i] = y[i]
            else:
                y[i] = self.last_good_y[i]
        
        # Safety override for TI-103
        safety_r = np.array(r)
        if y[2] < -0.5:
            safety_r[2] = max(r[2], -0.4)  # Force temperature recovery
        safety_r = np.nan_to_num(safety_r, nan=0.0)
        
        e = safety_r - y
        
        # Calculate control with anti-windup
        u = np.zeros(3)
        for i in range(3):
            self.integral[i] += self.Ki[i] * e[i] * self.sample_time
            u[i] = self.Kp[i] * e[i] + self.integral[i]
            
            # Clamping with anti-windup
            if u[i] > self.actuator_limits[1]:
                self.integral[i] -= (u[i] - self.actuator_limits[1])
                u[i] = self.actuator_limits[1]
            elif u[i] < self.actuator_limits[0]:
                self.integral[i] -= (u[i] - self.actuator_limits[0])
                u[i] = self.actuator_limits[0]

        # Gentle slew rate limiting (7.5%/sec)
        max_delta = 0.075 * self.sample_time
        u = np.clip(u, self.prev_u - max_delta, self.prev_u + max_delta)
        self.prev_u = u
        
        return np.clip(u, *self.actuator_limits)
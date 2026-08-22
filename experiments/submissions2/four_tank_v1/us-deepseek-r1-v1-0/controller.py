import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.Kp = [1.2, 1.2]
        self.Ki = [0.08, 0.08]
        self.reset()
        
    def reset(self):
        self.integral = [3.0, 3.0]
        self.prev_measurement = [None, None]
        self.first_step = True

    def step(self, t, y, r, quality):
        dt = self.sample_time
        u = np.zeros(2)
        
        for i in [0, 1]:
            if self.prev_measurement[i] is None:
                current_y = y[i] if quality[i] else r[i]
                self.prev_measurement[i] = current_y
                error = r[i] - current_y
                self.integral[i] = 3.0 - self.Kp[i] * error
            else:
                current_y = y[i] if quality[i] else self.prev_measurement[i]
                error = r[i] - current_y
                self.prev_measurement[i] = current_y
            
            p_term = self.Kp[i] * error
            u_unclamped = p_term + self.integral[i]
            u_clamped = np.clip(u_unclamped, 0.0, 10.0)
            
            # Anti-windup back calculation
            aw_correction = (u_clamped - u_unclamped)/self.Kp[i] if self.Kp[i] != 0 else 0.0
            self.integral[i] += (self.Ki[i] * error + aw_correction) * dt
            
            u[i] = u_clamped
            
        return u
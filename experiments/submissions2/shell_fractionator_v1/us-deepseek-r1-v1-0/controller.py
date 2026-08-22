import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp = [0.2, 0.3, 0.8]
        self.Ki = [0.004, 0.005, 0.1]
        self.Kd = [0.0, 0.0, 0.0]
        self.alpha_deriv = 0.3
        self.safety_margin = 0.05
        self.safety_kp = 2.0
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.filtered_deriv = np.zeros(3)
        self.last_good_y = np.zeros(3)
        self.prev_u = np.zeros(3)

    def step(self, t, y, r, quality):
        u = np.zeros(3)
        dt = self.brief.sample_time
        
        for i in range(3):
            if quality[i]:
                self.last_good_y[i] = y[i]
            current_y = self.last_good_y[i]
            setpoint = r[i] if not np.isnan(r[i]) else 0.0
            
            error = setpoint - current_y
            p = self.Kp[i] * error
            
            raw_deriv = (error - self.prev_error[i]) / dt
            self.filtered_deriv[i] = self.alpha_deriv * raw_deriv + (1 - self.alpha_deriv) * self.filtered_deriv[i]
            d = self.Kd[i] * self.filtered_deriv[i]
            
            unclamped = p + self.integral[i] + d
            clamped = np.clip(unclamped, -0.5, 0.5)
            
            self.integral[i] = clamped - p - d
            self.prev_error[i] = error
            u[i] = clamped

        current_y2 = self.last_good_y[2]
        safety_set = -0.5 + self.safety_margin
        if current_y2 < safety_set:
            safety_err = safety_set - current_y2
            u[2] += self.safety_kp * safety_err
            u[2] = np.clip(u[2], -0.5, 0.5)
            self.integral[2] = u[2] - (self.Kp[2] * (r[2] - current_y2) + self.filtered_deriv[2] * self.Kd[2])

        for i in range(3):
            delta = u[i] - self.prev_u[i]
            if abs(delta) > 0.016:
                u[i] = self.prev_u[i] + np.sign(delta) * 0.016
        self.prev_u = u.copy()
        
        return u
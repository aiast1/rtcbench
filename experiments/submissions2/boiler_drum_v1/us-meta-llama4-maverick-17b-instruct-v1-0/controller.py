import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp = 0.05  
        self.Ki = 0.005  
        self.Kd = 0.0  
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_level = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_level = None

    def step(self, t, y, r, quality):
        level = y[0]
        if not quality[0]:
            if self.previous_level is not None:
                level = self.previous_level
            else:
                level = 0.0  

        self.previous_level = level

        error = r[0] - level
        self.integral += error * self.brief.sample_time
        derivative = (error - self.previous_error) / self.brief.sample_time if self.brief.sample_time > 0 else 0.0
        self.previous_error = error

        self.integral = np.clip(self.integral, -10.0, 10.0)

        control_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        control_output = np.clip(control_output, 0.0, 100.0)  

        if t == 0:
            control_output = 45.0  # Initial actuator value as given in operating data

        return np.array([control_output])
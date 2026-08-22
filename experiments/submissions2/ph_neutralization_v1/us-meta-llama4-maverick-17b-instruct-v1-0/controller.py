import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp = -0.2  
        self.Ki = -0.02  
        self.Kd = -0.01  
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_pH = None
        self.u = np.array([14.22])  

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_pH = None
        self.u = np.array([14.22])

    def step(self, t, y, r, quality):
        if not quality[0]:
            return self.u  

        pH = y[0]
        level = y[1]
        setpoint = r[0]

        error = setpoint - pH

        self.integral += error * self.brief.sample_time
        if self.previous_pH is not None:
            derivative = (pH - self.previous_pH) / self.brief.sample_time
        else:
            derivative = 0.0
        self.previous_pH = pH

        delta_u = self.Kp * error + self.Ki * self.integral - self.Kd * derivative
        self.u += delta_u  

        if self.u[0] < 0.0:
            self.u[0] = 0.0
            if error > 0:
                self.integral -= error * self.brief.sample_time
        elif self.u[0] > 30.0:
            self.u[0] = 30.0
            if error < 0:
                self.integral -= error * self.brief.sample_time

        # More conservative control near level limits
        if level < 7.0 and self.u[0] < 5.0:
            self.u[0] = 5.0
        elif level > 28.0 and self.u[0] > 25.0:
            self.u[0] = 25.0

        self.previous_error = error
        return self.u
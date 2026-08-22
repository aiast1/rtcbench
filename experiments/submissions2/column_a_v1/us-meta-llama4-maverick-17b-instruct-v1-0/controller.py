import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp = np.array([[-0.1, 0.05], [0.05, -0.1]])
        self.Ki = np.array([[-0.0005, 0.0], [0.0, -0.0005]])
        self.Kd = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.previous_y = np.zeros(2)
        self.u = np.array([2.70629, 3.20629])

    def reset(self):
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.previous_y = np.zeros(2)
        self.u = np.array([2.70629, 3.20629])

    def step(self, t, y, r, quality):
        if not np.all(quality):
            return self.u
        
        error = r - y
        self.integral += self.Ki * error * self.brief.sample_time
        derivative = (y - self.previous_y) / self.brief.sample_time
        
        u_desired = np.clip(self.u + np.dot(self.Kp, error) + self.integral - np.dot(self.Kd, derivative), [1.5, 2.0], [4.5, 5.0])
        delta_u = u_desired - self.u
        
        # Limit delta_u to prevent excessive movement
        delta_u = np.clip(delta_u, -0.003, 0.003)
        
        self.u += delta_u
        self.previous_error = error
        self.previous_y = y
        
        return self.u
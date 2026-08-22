import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_u = 2  # two actuators
        self.n_y = 2  # two measured outputs
        
        # Controller gains (tuned for robustness and performance)
        self.Kp = np.array([0.08, 0.08])   # proportional gain
        self.Ki = np.array([0.003, 0.003]) # integral gain
        self.Kd = np.array([0.0, 0.0])     # derivative gain (set to zero due to noise and delay)
        
        # State
        self.integral = np.zeros(self.n_u)
        self.prev_error = np.zeros(self.n_y)
        self.prev_u = np.array([3.0, 3.0])  # bumpless start
        
        # Anti-windup limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Derivative filter (if needed, but Kd=0 so not used)
        self.alpha = 0.1  # derivative filter coefficient (not used if Kd=0)
        
    def reset(self):
        self.integral = np.zeros(self.n_u)
        self.prev_error = np.zeros(self.n_y)
        self.prev_u = np.array([3.0, 3.0])
        
    def step(self, t, y, r, quality):
        # Only use measurements that are good
        valid = quality & ~np.isnan(r)
        if not np.any(valid):
            # No valid measurements, hold previous control
            u = self.prev_u
        else:
            # Compute error only for valid channels
            error = np.zeros(self.n_y)
            error[valid] = r[valid] - y[valid]
            
            # Integral term (only update if error is valid)
            self.integral += error * self.sample_time
            
            # Anti-windup: clamp integral if output is saturated
            u_unsat = self.Kp * error + self.Ki * self.integral
            for i in range(self.n_u):
                if u_unsat[i] < self.u_min[i]:
                    self.integral[i] -= (u_unsat[i] - self.u_min[i]) / self.Ki[i] if self.Ki[i] != 0 else 0
                elif u_unsat[i] > self.u_max[i]:
                    self.integral[i] -= (u_unsat[i] - self.u_max[i]) / self.Ki[i] if self.Ki[i] != 0 else 0
            
            # Recompute control with anti-windup
            u = self.Kp * error + self.Ki * self.integral
            
            # Derivative term (if Kd were non-zero, but it's zero)
            # derivative = (error - self.prev_error) / self.sample_time
            # u += self.Kd * derivative
            # self.prev_error = error
            
        # Actuator rate limiting (slew rate) to prevent chattering and respect actuator limits
        max_delta = 0.017 * self.sample_time  # max change per step from duty limit
        delta = u - self.prev_u
        delta = np.clip(delta, -max_delta, max_delta)
        u = self.prev_u + delta
        
        # Final clamp to actuator limits
        u = np.clip(u, self.u_min, self.u_max)
        
        # Store for next step
        self.prev_u = u.copy()
        
        return u
import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.n_meas = len(brief.measurements)
        self.n_act = len(brief.actuators)
        
        # Initialize PID gains
        self.Kp_CB = 0.1
        self.Ki_CB = 0.01
        self.Kd_CB = 0.001
        self.Kp_T = 0.1
        self.Ki_T = 0.01
        self.Kd_T = 0.001
        
        # Initialize integral terms
        self.integral_CB = 0.0
        self.integral_T = 0.0
        
        # Initialize previous measurement and error
        self.prev_error_CB = 0.0
        self.prev_error_T = 0.0
        self.prev_meas_CB = None
        self.prev_meas_T = None
        self.prev_time = 0
        
        # Initialize actuator limits
        self.act_limits = np.array([[3.0, 35.0], [-9000.0, 0.0]])
        
        # Initialize setpoint schedule
        self.setpoint_schedule = np.array([
            [0, [1.09, 114.19]],
            [900, [0.928, 114.19]],
            [2900, [1.081, 114.19]],
            [4600, [0.892, 114.19]],
            [6100, [1.09, 114.19]]
        ])
        
        # Initialize current setpoint
        self.current_setpoint = np.array([1.09, 114.19])
        
        # Initialize actuator slew rate limit
        self.act_slew_limit = 0.00134 / self.sample_time
        
        # Initialize previous actuator values
        self.prev_act = np.array([14.19, -1113.5])
        
    def reset(self):
        self.integral_CB = 0.0
        self.integral_T = 0.0
        self.prev_error_CB = 0.0
        self.prev_error_T = 0.0
        self.prev_meas_CB = None
        self.prev_meas_T = None
        self.prev_time = 0
        self.prev_act = np.array([14.19, -1113.5])
        
    def step(self, t, y, r, quality):
        # Update current setpoint
        for interval, setpoint in self.setpoint_schedule:
            if t >= interval:
                self.current_setpoint = np.array(setpoint)
        
        # Check quality flags
        if not quality[0]:
            y[0] = self.prev_meas_CB
        if not quality[1]:
            y[1] = self.prev_meas_T
        
        # Update previous measurements
        self.prev_meas_CB = y[0]
        self.prev_meas_T = y[1]
        
        # Calculate errors
        error_CB = self.current_setpoint[0] - y[0]
        error_T = self.current_setpoint[1] - y[1]
        
        # Update integral terms with anti-windup
        self.integral_CB += self.sample_time * error_CB
        self.integral_T += self.sample_time * error_T
        
        # Anti-windup for integral terms
        act_CB = np.clip(self.prev_act[0], self.act_limits[0, 0], self.act_limits[0, 1])
        act_T = np.clip(self.prev_act[1], self.act_limits[1, 0], self.act_limits[1, 1])
        if act_CB == self.act_limits[0, 0] or act_CB == self.act_limits[0, 1]:
            self.integral_CB -= self.sample_time * error_CB
        if act_T == self.act_limits[1, 0] or act_T == self.act_limits[1, 1]:
            self.integral_T -= self.sample_time * error_T
        
        # Calculate derivatives
        if self.prev_time > 0:
            deriv_CB = (error_CB - self.prev_error_CB) / (t - self.prev_time)
            deriv_T = (error_T - self.prev_error_T) / (t - self.prev_time)
        else:
            deriv_CB = 0.0
            deriv_T = 0.0
        
        # Update previous errors and time
        self.prev_error_CB = error_CB
        self.prev_error_T = error_T
        self.prev_time = t
        
        # Calculate PID outputs
        pid_CB = self.Kp_CB * error_CB + self.Ki_CB * self.integral_CB + self.Kd_CB * deriv_CB
        pid_T = self.Kp_T * error_T + self.Ki_T * self.integral_T + self.Kd_T * deriv_T
        
        # Convert PID outputs to actuator values
        act_CB = np.clip(pid_CB + 14.19, self.act_limits[0, 0], self.act_limits[0, 1])
        act_T = np.clip(pid_T - 1113.5, self.act_limits[1, 0], self.act_limits[1, 1])
        
        # Apply actuator slew rate limit
        act_CB = np.clip(act_CB, self.prev_act[0] - self.act_slew_limit, self.prev_act[0] + self.act_slew_limit)
        act_T = np.clip(act_T, self.prev_act[1] - self.act_slew_limit, self.prev_act[1] + self.act_slew_limit)
        
        # Ensure safe temperature
        if y[1] > 150.0:
            act_T = -9000.0
        
        self.prev_act = np.array([act_CB, act_T])
        
        return self.prev_act
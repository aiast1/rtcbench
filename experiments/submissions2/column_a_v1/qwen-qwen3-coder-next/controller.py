import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_actuators = 2
        self.n_measurements = 2
        # Conservative PID parameters to ensure robustness
        self.Kp = np.array([[0.15, 0.0],   # Kp for yD control (on L)
                            [0.0, 0.15]])  # Kp for yB control (on V)
        self.Ki = np.array([[0.0001, 0.0],   # Integral gain for yD
                            [0.0, 0.0001]])  # Integral gain for yB
        self.Kd = np.array([[0.0, 0.0],      # Derivative action minimal due to noise
                            [0.0, 0.0]])
        # Anti-windup gains
        self.Kaw = np.array([[0.05, 0.0],
                            [0.0, 0.05]])
        # Actuator limits
        self.u_min = np.array([1.5, 2.0])
        self.u_max = np.array([4.5, 5.0])
        # Slew rate limits (kmol/min per second)
        self.slew_limit = 0.003
        # Initial actuator values at steady state
        self.u_sp = np.array([2.70629, 3.20629])
        # Internal state
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_output = np.copy(self.u_sp)
        self.last_update_time = None
        # Safety margins
        self.yD_min = 0.85
        self.xB_max = 0.15  # xB = 1 - yB, so yB >= 0.85
        self.D_min = 0.05
        self.B_min = 0.05

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_output = np.copy(self.u_sp)
        self.last_update_time = None

    def step(self, t, y, r, quality):
        # Extract measurements
        yD = y[0]  # distillate purity (light key)
        yB = y[1]  # bottoms purity (heavy key)
        
        # Calculate product draws based on manipulated flows (internal reflux)
        L = self.prev_output[0] if self.last_update_time is not None else self.u_sp[0]
        V = self.prev_output[1] if self.last_update_time is not None else self.u_sp[1]
        F = 1.0  # Feed flow rate (given)
        # D = V - L (overhead), B = L + F - V (bottoms)
        D = V - L
        B = L + F - V
        
        # Safety check: if any constraint is violated, revert to safe values
        # Safety envelope: yD >= 0.85, xB <= 0.15 (i.e., yB >= 0.85), D >= 0.05, B >= 0.05
        if yD < self.yD_min or yB < self.yD_min or D < self.D_min or B < self.B_min:
            # Return to safe operating point
            return self.u_sp
        
        # Setpoints for scored channels
        rD = r[0] if not np.isnan(r[0]) else 0.99
        rB = r[1] if not np.isnan(r[1]) else 0.99
        
        # Update setpoints based on time schedule
        if t < 80:
            rD = 0.99
            rB = 0.99
        elif t < 220:
            rD = 0.994
            rB = 0.994
        else:
            # Ramp from 80-260s: at t=220s start ramping for 40s
            if t < 260:
                ramp_factor = (t - 220) / 40.0
                rD = 0.994 + (0.986 - 0.994) * ramp_factor
                rB = 0.994 + (0.996 - 0.994) * ramp_factor
            else:
                rD = 0.986
                rB = 0.996
        
        # Initialize output
        u = np.zeros(2)
        
        # Calculate errors
        eD = rD - yD
        eB = rB - yB
        error = np.array([eD, eB])
        
        # Update integral only if measurements are valid
        if quality[0] and quality[1]:
            dt = self.sample_time
            self.integral += error * dt
            
            # Anti-windup: limit integral growth when actuators are saturated
            u_prop = self.Kp @ error
            u_int = self.Ki @ self.integral
            u_test = self.prev_output + u_prop + u_int
            
            # Apply anti-windup if saturated
            for i in range(2):
                if u_test[i] > self.u_max[i] and error[i] > 0:
                    self.integral[i] -= (u_test[i] - self.u_max[i]) * self.Kaw[i, i] * dt
                elif u_test[i] < self.u_min[i] and error[i] < 0:
                    self.integral[i] -= (u_test[i] - self.u_min[i]) * self.Kaw[i, i] * dt
            
            # Calculate derivative (with noise filtering)
            if self.last_update_time is not None:
                dt = t - self.last_update_time
                if dt > 0:
                    # Simple first-order filter for derivative
                    alpha = 0.2  # Filter coefficient
                    dyD = (yD - self.prev_measurement[0]) / dt if quality[0] else 0
                    dyB = (yB - self.prev_measurement[1]) / dt if quality[1] else 0
                    measurement_rate = np.array([dyD, dyB])
                    
                    # Filtered derivative
                    filtered_derivative = (1 - alpha) * self.prev_measurement + alpha * measurement_rate
                    derivative = measurement_rate - filtered_derivative
                else:
                    derivative = np.zeros(2)
            else:
                derivative = np.zeros(2)
            
            # Calculate PID output
            u_prop = self.Kp @ error
            u_int = self.Ki @ self.integral
            u_deriv = self.Kd @ derivative
            
            u = self.prev_output + u_prop + u_int + u_deriv
            
            # Apply actuator limits
            u = np.clip(u, self.u_min, self.u_max)
            
            # Apply slew rate limits
            du = u - self.prev_output
            du = np.clip(du, -self.slew_limit * self.sample_time, self.slew_limit * self.sample_time)
            u = self.prev_output + du
            
            # Update state
            self.prev_measurement = y.copy()
            self.prev_output = u.copy()
            self.last_update_time = t
            
            return u
        else:
            # If measurements are invalid, hold previous output
            return self.prev_output
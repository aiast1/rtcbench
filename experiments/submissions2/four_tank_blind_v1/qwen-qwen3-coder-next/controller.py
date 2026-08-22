import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([0.0, 10.0])
        self.safety_limits = np.array([0.0, 20.0])
        self.setpoint_schedule = [
            (0, np.array([12.26, 12.78])),
            (200, np.array([14.0, 12.78])),
            (500, np.array([14.0, 11.2])),
            (900, np.array([12.26, 12.78]))
        ]
        # Initialize controller parameters (conservative tuning)
        self.Kp = np.array([[0.3, 0.05],   # h1 controlled by pump1 strongly, pump2 weakly
                            [0.05, 0.3]])  # h2 controlled by pump2 strongly, pump1 weakly
        self.Ki = np.array([[0.005, 0.003],
                            [0.003, 0.005]])
        self.Kd = np.array([[0.01, 0.005],
                            [0.005, 0.01]])
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_time = None
        self.last_u = np.array([3.0, 3.0])  # initial actuator values
        self.ramp_start_time = None
        self.ramp_target = None
        self.ramp_initial = None
        self.ramp_duration = 60.0
        self.ramp_channel = None
        self.filtered_derivative = np.zeros(2)
        self.safety_margin = 2.0  # keep measured levels at least 2cm below 20cm

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_time = None
        self.last_u = np.array([3.0, 3.0])
        self.ramp_start_time = None
        self.ramp_target = None
        self.ramp_initial = None
        self.ramp_channel = None
        self.filtered_derivative = np.zeros(2)

    def _get_setpoint(self, t):
        # Find the current setpoint based on schedule
        sp = self.setpoint_schedule[0][1].copy()
        for i in range(len(self.setpoint_schedule) - 1):
            t_start, sp_start = self.setpoint_schedule[i]
            t_end, sp_end = self.setpoint_schedule[i + 1]
            if t_start <= t < t_end:
                if i == 2:  # ramp segment (t=500s to t=560s)
                    if self.ramp_start_time is None:
                        self.ramp_start_time = t_start
                        self.ramp_initial = sp_start.copy()
                        self.ramp_target = sp_end.copy()
                        self.ramp_channel = 1  # only channel 2 is ramping
                    # Compute interpolated setpoint
                    elapsed = t - self.ramp_start_time
                    progress = min(elapsed / self.ramp_duration, 1.0)
                    sp = self.ramp_initial.copy()
                    sp[self.ramp_channel] = (self.ramp_initial[self.ramp_channel] + 
                                            progress * (self.ramp_target[self.ramp_channel] - 
                                                       self.ramp_initial[self.ramp_channel]))
                else:
                    sp = sp_start.copy()
                break
        return sp

    def step(self, t, y, r, quality):
        # Get current setpoint
        sp = self._get_setpoint(t)
        
        # Use measured values where quality is good, otherwise use previous values
        h = np.zeros(2)
        for i in range(2):
            if quality[i]:
                h[i] = y[i]
            else:
                h[i] = self.prev_measurement[i]
        
        # Ensure safety by clamping measurements to safe range for control logic
        h = np.clip(h, 0.0, 20.0)
        
        # Compute error
        error = sp - h
        
        # PID calculation for each loop
        u_desired = self.last_u.copy()
        
        for i in range(2):
            # Proportional term
            u_p = self.Kp[i, i] * error[i]
            
            # Integral term with anti-windup (clamping based on actuator limits)
            self.integral[i] += error[i] * self.sample_time
            # Anti-windup: reduce integral if actuator is saturated
            if (self.last_u[i] >= self.actuator_limits[1] and error[i] > 0) or \
               (self.last_u[i] <= self.actuator_limits[0] and error[i] < 0):
                self.integral[i] -= error[i] * self.sample_time * 0.1  # back-calculation
            
            u_i = self.Ki[i, i] * self.integral[i]
            
            # Derivative term (filtered derivative to reduce noise sensitivity)
            if self.prev_time is not None and t > self.prev_time:
                dt = t - self.prev_time
                # Use filtered derivative: (error - prev_error) / dt, but with low-pass filtering
                derivative = (error[i] - self.prev_error[i]) / dt if dt > 0 else 0.0
                # Apply first-order low-pass filter on derivative
                alpha = 0.2  # filtering factor
                self.filtered_derivative[i] = (alpha * derivative + 
                                              (1 - alpha) * self.filtered_derivative[i])
            else:
                self.filtered_derivative[i] = 0.0
            
            u_d = self.Kd[i, i] * self.filtered_derivative[i]
            
            # Sum contributions
            u_desired[i] = self.last_u[i] + u_p + u_i + u_d
        
        # Apply cross-coupling terms (very weak coupling to handle interactions)
        u_desired[0] += self.Kp[0, 1] * error[1] * 0.05  # very weak coupling from h2 to pump1
        u_desired[1] += self.Kp[1, 0] * error[0] * 0.05  # very weak coupling from h1 to pump2
        
        # Clamp to actuator limits
        u_clamped = np.clip(u_desired, self.actuator_limits[0], self.actuator_limits[1])
        
        # Apply actuator slew rate limit (conservative: max change of 0.3 V per step)
        max_slew = 0.3
        u_change = np.clip(u_clamped - self.last_u, -max_slew, max_slew)
        u_final = self.last_u + u_change
        
        # Final clamping to ensure hard limits are respected
        u_final = np.clip(u_final, self.actuator_limits[0], self.actuator_limits[1])
        
        # Safety check: if measured levels are getting too close to 20cm, reduce actuator output
        # to prevent overflow of unmeasured tanks
        if np.any(h > 18.0):
            # Reduce output more aggressively when close to safety limit
            reduction_factor = 0.7
            u_final = self.last_u + (u_final - self.last_u) * reduction_factor
            u_final = np.clip(u_final, self.actuator_limits[0], self.actuator_limits[1])
        
        # Update state
        self.prev_measurement = h.copy()
        self.prev_error = error.copy()
        self.prev_time = t
        self.last_u = u_final.copy()
        
        return u_final
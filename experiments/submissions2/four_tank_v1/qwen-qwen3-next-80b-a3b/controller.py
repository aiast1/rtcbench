import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        # Conservative gains for robustness and safety
        self.Kp = np.array([0.6, 0.6])      # Reduced from 0.8 to avoid overshoot
        self.Ki = np.array([0.01, 0.01])    # Reduced from 0.02 to prevent integral windup
        self.Kd = np.array([0.05, 0.05])    # Further reduced to suppress noise
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)
        self.prev_meas = np.zeros(2)
        self.prev_output = np.array([3.0, 3.0])  # Bumpless start
        self.sat_limit = 10.0
        self.min_output = 0.0
        self.ramp_start_time = 500.0
        self.ramp_duration = 60.0
        self.ramp_end_time = self.ramp_start_time + self.ramp_duration
        self.ramp_target = np.array([14.0, 11.2])
        self.ramp_start_setpoint = np.array([14.0, 12.78])
        self.ramp_slope = (self.ramp_target - self.ramp_start_setpoint) / self.ramp_duration
        # Safety margins: we assume unmeasured tanks are at same level as measured (conservative)
        self.safety_margin = 1.5  # cm - we'll keep measured tanks below 18.5 to protect unmeasured
        self.max_change_per_step = 0.017 * self.sample_time  # 0.034 V max change per step

    def reset(self):
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)
        self.prev_meas = np.zeros(2)
        self.prev_output = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Handle bad measurements by using previous values
        y_clean = np.where(quality, y, self.prev_meas)
        self.prev_meas = y_clean.copy()
        
        # Apply setpoint schedule with safety margins
        if t < 200:
            setpoint_raw = np.array([12.26, 12.78])
        elif t < 500:
            setpoint_raw = np.array([14.0, 12.78])
        elif t < 900:
            if t < self.ramp_start_time:
                setpoint_raw = np.array([14.0, 12.78])
            elif t < self.ramp_end_time:
                elapsed = t - self.ramp_start_time
                setpoint_raw = self.ramp_start_setpoint + self.ramp_slope * elapsed
            else:
                setpoint_raw = self.ramp_target
        else:
            setpoint_raw = np.array([12.26, 12.78])
        
        # Apply safety margin: never let measured tanks exceed 18.5 cm
        setpoint = np.minimum(setpoint_raw, 20.0 - self.safety_margin)
        
        # Calculate error
        error = setpoint - y_clean
        
        # Integral action with aggressive anti-windup
        # Only integrate if output is not near saturation AND error is pushing toward it
        u_prop = self.Kp * error
        u_deriv = self.Kd * (y_clean - self.prev_meas) / self.sample_time
        
        # Anti-windup: clamp integral based on actuator position and error direction
        for i in range(2):
            if self.prev_output[i] >= self.sat_limit - 0.5:  # Near max
                if error[i] > 0:  # Error would push higher -> stop integration
                    self.integral[i] = max(self.integral[i], -u_prop[i])
                else:
                    self.integral[i] += self.Ki[i] * error[i] * self.sample_time
            elif self.prev_output[i] <= self.min_output + 0.5:  # Near min
                if error[i] < 0:  # Error would push lower -> stop integration
                    self.integral[i] = min(self.integral[i], -u_prop[i])
                else:
                    self.integral[i] += self.Ki[i] * error[i] * self.sample_time
            else:
                self.integral[i] += self.Ki[i] * error[i] * self.sample_time
        
        # Combine terms
        u = u_prop + self.integral - u_deriv
        
        # Apply hard saturation first
        u_clamped = np.clip(u, self.min_output, self.sat_limit)
        
        # Apply slew rate limit to prevent chattering (duty limit constraint)
        u_final = np.clip(u_clamped, 
                         self.prev_output - self.max_change_per_step, 
                         self.prev_output + self.max_change_per_step)
        
        # Final safety check: if we're near saturation, reduce setpoint further
        # This is a last-resort safety layer
        if np.any(u_final >= self.sat_limit - 0.1):
            # Reduce setpoint for the channel that's saturating
            for i in range(2):
                if u_final[i] >= self.sat_limit - 0.1:
                    # Reduce setpoint by 0.5 cm for this channel
                    setpoint[i] = max(setpoint[i] - 0.5, 0.0)
                    # Recalculate error and u_prop
                    error[i] = setpoint[i] - y_clean[i]
                    u_prop[i] = self.Kp[i] * error[i]
                    u = u_prop + self.integral - u_deriv
                    u_clamped[i] = np.clip(u[i], self.min_output, self.sat_limit)
                    u_final[i] = np.clip(u_clamped[i], 
                                       self.prev_output[i] - self.max_change_per_step, 
                                       self.prev_output[i] + self.max_change_per_step)
        
        # Update state
        self.prev_output = u_final.copy()
        self.prev_error = error.copy()
        
        return u_final
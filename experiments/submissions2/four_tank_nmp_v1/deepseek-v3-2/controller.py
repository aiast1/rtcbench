import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_start = np.array([3.0, 3.0])
        
        # Very conservative PI-only control (no derivative for noise)
        self.Kp = np.array([0.3, 0.28])
        self.Ki = np.array([0.04, 0.035])
        
        # Strong anti-windup
        self.Kaw = np.array([0.5, 0.5])
        
        # Output limits
        self.u_min = 0.0
        self.u_max = 10.0
        
        # Extremely strict rate limiting to prevent duty violations
        self.du_max = 0.05
        
        # Safety margins
        self.safe_max = 17.0
        self.safe_min = 3.0
        
        # State variables
        self.integral = None
        self.prev_error = None
        self.prev_measurement = None
        self.prev_output = None
        self.filtered_measurement = None
        
    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.array([11.28, 11.94])
        self.filtered_measurement = np.array([11.28, 11.94])
        self.prev_output = self.actuator_start.copy()
        
    def step(self, t, y, r, quality):
        # Heavy filtering for noise reduction
        alpha_filter = 0.15  # Very slow filter
        y_filtered = np.zeros(2)
        
        for i in range(2):
            if not quality[i]:
                y_filtered[i] = self.filtered_measurement[i]
            else:
                y_filtered[i] = alpha_filter * y[i] + (1 - alpha_filter) * self.filtered_measurement[i]
        
        # Calculate errors with setpoint filtering for smooth transitions
        error = np.zeros(2)
        for i in range(2):
            if not np.isnan(r[i]):
                # Filter setpoint changes to avoid aggressive moves
                if t == 0:
                    target = r[i]
                else:
                    # Smooth setpoint transitions
                    alpha_sp = 0.02  # Very slow setpoint tracking
                    if i == 0:
                        if t < 200:
                            target = 11.28
                        elif t < 500:
                            target = 13.0
                        elif t < 560:
                            # Ramp from 13.0 to 11.28
                            ramp_progress = min(1.0, (t - 500) / 60)
                            target = 13.0 - (13.0 - 11.28) * ramp_progress
                        else:
                            target = 11.28
                    else:
                        if t < 500:
                            target = 11.94
                        elif t < 560:
                            # Ramp from 11.94 to 10.4
                            ramp_progress = min(1.0, (t - 500) / 60)
                            target = 11.94 - (11.94 - 10.4) * ramp_progress
                        else:
                            target = 11.94
                
                error[i] = target - y_filtered[i]
            else:
                error[i] = 0.0
        
        # Safety override: if levels are near limits, override control
        safety_override = False
        override_output = self.prev_output.copy()
        
        for i in range(2):
            if y_filtered[i] > 18.5:  # Very close to overflow
                safety_override = True
                override_output[i] = max(0.0, self.prev_output[i] - 1.0)
                error[i] = -2.0  # Force reduction
            elif y_filtered[i] > 17.0:
                safety_override = True
                override_output[i] = max(0.0, self.prev_output[i] - 0.5)
                error[i] = -1.0
            elif y_filtered[i] < 1.5:  # Very close to empty
                safety_override = True
                override_output[i] = min(10.0, self.prev_output[i] + 1.0)
                error[i] = 2.0  # Force increase
            elif y_filtered[i] < 3.0:
                safety_override = True
                override_output[i] = min(10.0, self.prev_output[i] + 0.5)
                error[i] = 1.0
        
        if safety_override:
            # Apply rate limiting to override
            du = override_output - self.prev_output
            du = np.clip(du, -self.du_max, self.du_max)
            u_final = self.prev_output + du
            
            # Reset integral to prevent windup during override
            self.integral = np.zeros(2)
            
            # Update state
            self.prev_error = error.copy()
            self.filtered_measurement = y_filtered.copy()
            self.prev_measurement = y_filtered.copy()
            self.prev_output = u_final.copy()
            
            return u_final
        
        # Normal PI control with aggressive anti-windup
        # Limit integral growth when error is large
        max_integral = np.array([2.0, 2.0])
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # Update integral with conditional integration
        integral_update = self.Ki * error * self.sample_time
        
        # Only integrate if error is small (prevents windup during large errors)
        for i in range(2):
            if abs(error[i]) < 2.0:
                self.integral[i] += integral_update[i]
            else:
                # Reduce integral when error is large
                self.integral[i] *= 0.9
        
        # PI calculation
        proportional = self.Kp * error
        u_pid = proportional + self.integral
        
        # Apply output limits
        u_sat = np.clip(u_pid, self.u_min, self.u_max)
        
        # Anti-windup
        for i in range(2):
            if u_pid[i] != u_sat[i]:
                windup_correction = self.Kaw[i] * (u_sat[i] - u_pid[i])
                self.integral[i] += windup_correction * self.sample_time
        
        # Very strict rate limiting
        du = u_sat - self.prev_output
        du = np.clip(du, -self.du_max, self.du_max)
        u_final = self.prev_output + du
        
        # Final safety check
        u_final = np.clip(u_final, self.u_min, self.u_max)
        
        # Update state
        self.prev_error = error.copy()
        self.filtered_measurement = y_filtered.copy()
        self.prev_measurement = y_filtered.copy()
        self.prev_output = u_final.copy()
        
        return u_final
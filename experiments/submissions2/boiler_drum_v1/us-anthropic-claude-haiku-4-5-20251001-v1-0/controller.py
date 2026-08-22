import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # State for integral action and anti-windup
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_output = 45.0  # Start at steady state
        self.last_measurement = 0.0
        
        # Conservative PID gains
        self.Kp = 0.2
        self.Ki = 0.03
        self.Kd = 0.01
        
        # Anti-windup limits
        self.integral_max = 10.0
        self.integral_min = -10.0
        
        # Low-pass filter for derivative
        self.alpha = 0.15
        self.filtered_derivative = 0.0
        
        # Valve saturation limits
        self.valve_min = 0.0
        self.valve_max = 100.0
        
        # Rate limiting - very conservative to avoid duty violations
        self.max_rate = 0.2  # % per sample
        
        # Dead band to reduce chatter
        self.dead_band = 1.5  # mm
        
        # Track if we're in a setpoint change to be more aggressive during transitions
        self.last_setpoint = 0.0
        self.setpoint_change_time = 0
        self.transition_window = 60  # seconds to be more aggressive
        
    def reset(self):
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_output = 45.0
        self.last_measurement = 0.0
        self.filtered_derivative = 0.0
        self.last_setpoint = 0.0
        self.setpoint_change_time = 0
    
    def step(self, t, y, r, quality):
        """
        Control the feedwater valve to track drum level setpoint.
        
        Strategy:
        1. Use very conservative rate limiting to stay within duty cycle
        2. Detect setpoint changes and allow slightly more aggressive response
        3. Use integral action to remove steady-state offset
        4. Minimize derivative action due to noise and delay
        5. Implement strict anti-windup
        """
        
        # Extract measurements
        l_ind = y[0]  # Indicated level (mm)
        p_drum = y[1]  # Drum pressure (bar)
        f_steam = y[2]  # Steam flow (kg/s)
        f_feedwater = y[3]  # Feedwater flow (kg/s)
        
        # Get setpoint for indicated level
        l_setpoint = r[0]
        
        # Detect setpoint change
        if not np.isnan(l_setpoint) and l_setpoint != self.last_setpoint:
            self.setpoint_change_time = t
            self.last_setpoint = l_setpoint
        
        # Check if we're in transition window
        in_transition = (t - self.setpoint_change_time) < self.transition_window
        
        # Use quality flags to decide whether to trust measurements
        if not quality[0]:
            measurement = self.last_measurement
        else:
            measurement = l_ind
            self.last_measurement = l_ind
        
        # Calculate error
        error = l_setpoint - measurement
        
        # Apply dead band only when not in transition
        if not in_transition and abs(error) < self.dead_band:
            error = 0.0
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term with strict anti-windup
        if not (self.last_output >= self.valve_max and error > 0) and \
           not (self.last_output <= self.valve_min and error < 0):
            self.integral_error += error * self.sample_time
            self.integral_error = np.clip(self.integral_error, 
                                         self.integral_min, 
                                         self.integral_max)
        else:
            # Decay integral when saturated
            self.integral_error *= 0.90
        
        i_term = self.Ki * self.integral_error
        
        # Derivative term - very conservative
        if quality[0]:
            raw_derivative = (error - self.last_error) / self.sample_time
            self.filtered_derivative = (self.alpha * raw_derivative + 
                                       (1.0 - self.alpha) * self.filtered_derivative)
            d_term = self.Kd * self.filtered_derivative
        else:
            d_term = 0.0
        
        self.last_error = error
        
        # Combine PID terms
        pid_output = p_term + i_term + d_term
        
        # Adaptive rate limiting: slightly more aggressive during transitions
        if in_transition:
            max_rate = 0.3  # % per sample during setpoint change
        else:
            max_rate = 0.2  # % per sample at steady state
        
        max_change = max_rate * self.sample_time / 5.0
        
        # Calculate desired output
        desired_output = self.last_output + pid_output
        
        # Apply rate limiting
        output = np.clip(desired_output, 
                        self.last_output - max_change,
                        self.last_output + max_change)
        
        # Apply valve saturation limits
        output = np.clip(output, self.valve_min, self.valve_max)
        
        self.last_output = output
        
        return np.array([output])
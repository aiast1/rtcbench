import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.Kp = 0.8
        self.Ki = 0.02
        self.Kd = 0.1
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.prev_output = 14.22
        self.output_limit = [0.0, 30.0]
        self.level_min = 5.0
        self.level_max = 30.0
        self.pH_min = 4.0
        self.pH_max = 10.5
        self.slew_limit = 0.0041 * self.sample_time  # max change per step
        self.delay_samples = int(20.0 / self.sample_time)  # 20s delay
        self.delay_buffer = np.zeros(self.delay_samples)
        self.setpoint_history = []
        self.last_setpoint = 6.5
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 6.5
        self.prev_output = 14.22
        self.delay_buffer = np.zeros(self.delay_samples)
        self.setpoint_history = []
        self.last_setpoint = 6.5

    def step(self, t, y, r, quality):
        # Extract measurements and setpoints
        pH_measured = y[0]
        level = y[1]
        setpoint = r[0]
        
        # Handle bad measurements
        if not quality[0] or np.isnan(pH_measured):
            pH_measured = self.prev_measurement
        if np.isnan(setpoint):
            setpoint = self.last_setpoint
        else:
            self.last_setpoint = setpoint
            
        # Update delay buffer (simulate 20s transport delay)
        self.delay_buffer = np.roll(self.delay_buffer, 1)
        self.delay_buffer[0] = pH_measured
        pH_delayed = self.delay_buffer[-1]
        
        # Calculate error
        error = setpoint - pH_delayed
        
        # Anti-windup: limit integral action based on actuator saturation
        if self.prev_output >= self.output_limit[1] and error > 0:
            # Output at max and error positive: don't integrate
            integral_term = self.integral
        elif self.prev_output <= self.output_limit[0] and error < 0:
            # Output at min and error negative: don't integrate
            integral_term = self.integral
        else:
            integral_term = self.integral + self.Ki * error * self.sample_time
            
        # Derivative term with filtering to reduce noise impact
        derivative = self.Kd * (pH_delayed - self.prev_measurement) / self.sample_time
        self.prev_measurement = pH_delayed
        
        # PID output
        output = self.prev_output + self.Kp * error + integral_term - derivative
        
        # Anti-windup: clamp integral to prevent windup
        if output > self.output_limit[1]:
            output = self.output_limit[1]
            integral_term = self.integral - self.Ki * error * self.sample_time
        elif output < self.output_limit[0]:
            output = self.output_limit[0]
            integral_term = self.integral - self.Ki * error * self.sample_time
            
        # Slew rate limiting
        output = np.clip(output, 
                         self.prev_output - self.slew_limit, 
                         self.prev_output + self.slew_limit)
        
        # Level-based safety constraint: if level is near limits, reduce output change
        if level <= self.level_min + 1.0 and output > self.prev_output:
            output = min(output, self.prev_output + self.slew_limit * 0.3)
        elif level >= self.level_max - 1.0 and output < self.prev_output:
            output = max(output, self.prev_output - self.slew_limit * 0.3)
            
        # Final output clamping
        output = np.clip(output, self.output_limit[0], self.output_limit[1])
        
        # Update state
        self.integral = integral_term
        self.prev_output = output
        self.prev_error = error
        
        return np.array([output])
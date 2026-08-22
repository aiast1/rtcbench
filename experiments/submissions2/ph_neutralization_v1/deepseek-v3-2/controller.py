import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        # Extract float values from Channel objects
        self.actuator_start = float(brief.actuators[0])
        self.actuator_limits = (0.0, 30.0)
        self.level_limits = (5.0, 30.0)
        self.pH_limits = (4.0, 10.5)
        
        # Conservative PID parameters for robust control
        self.Kp = 1.2
        self.Ki = 0.03
        self.Kd = 0.15
        
        # Filter parameters for noisy, delayed measurement
        self.alpha = 0.2  # Stronger low-pass filter
        self.filtered_pH = None
        self.prev_filtered_pH = None
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = self.actuator_start
        self.prev_pH = None
        self.last_good_pH = None
        
        # Anti-windup
        self.integral_limit = 8.0
        
        # Derivative filter
        self.derivative_filter = 0.0
        self.alpha_d = 0.3  # Stronger derivative filtering
        
        # Output rate limiting (conservative)
        self.max_rate = 0.002 * self.sample_time  # More conservative than duty limit
        
        # Buffer for handling transport delay
        self.pH_buffer = []
        self.buffer_size = max(4, int(20 / self.sample_time))  # ~20 seconds delay
        
        # Level safety margin
        self.level_margin = 3.0
        
    def reset(self):
        self.filtered_pH = None
        self.prev_filtered_pH = None
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = self.actuator_start
        self.prev_pH = None
        self.last_good_pH = None
        self.derivative_filter = 0.0
        self.pH_buffer = []
        
    def step(self, t, y, r, quality):
        # Extract float values from measurements
        pH_measurement = float(y[0])
        level_measurement = float(y[1])
        
        # Extract setpoint as float
        pH_setpoint = float(r[0]) if not np.isnan(r[0]) else 7.0
        
        # Handle bad quality readings
        if not quality[0] or np.isnan(pH_measurement):
            if self.last_good_pH is not None:
                pH_measurement = self.last_good_pH
            else:
                pH_measurement = 7.0  # Safer default value
        else:
            self.last_good_pH = pH_measurement
        
        # Store pH in buffer for delay compensation
        self.pH_buffer.append(pH_measurement)
        if len(self.pH_buffer) > self.buffer_size:
            self.pH_buffer.pop(0)
        
        # Use delayed pH if buffer is full, otherwise use current
        if len(self.pH_buffer) >= self.buffer_size:
            delayed_pH = np.mean(self.pH_buffer[:int(self.buffer_size/2)])  # Use older half of buffer
        else:
            delayed_pH = pH_measurement
        
        # Initialize filtered pH
        if self.filtered_pH is None:
            self.filtered_pH = delayed_pH
            self.prev_filtered_pH = delayed_pH
        
        # Apply low-pass filter to delayed pH measurement
        self.filtered_pH = self.alpha * delayed_pH + (1 - self.alpha) * self.prev_filtered_pH
        self.prev_filtered_pH = self.filtered_pH
        
        # Calculate error
        error = pH_setpoint - self.filtered_pH
        
        # Integral term with conditional integration and anti-windup
        # Only integrate when error is small to prevent windup during large disturbances
        if abs(error) < 2.0:  # Only integrate near setpoint
            self.integral += self.Ki * error * self.sample_time
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        else:
            # Reset integral when far from setpoint
            self.integral = 0.0
        
        # Derivative term with heavy filtering
        if self.prev_pH is not None:
            derivative = (self.filtered_pH - self.prev_pH) / self.sample_time
            self.derivative_filter = self.alpha_d * derivative + (1 - self.alpha_d) * self.derivative_filter
            derivative_term = -self.Kd * self.derivative_filter  # Negative because error = setpoint - measurement
        else:
            derivative_term = 0.0
        
        self.prev_pH = self.filtered_pH
        
        # Calculate PID output
        output = self.Kp * error + self.integral + derivative_term
        
        # Add base output for steady-state (bumpless transfer)
        output += self.actuator_start
        
        # Apply level-based safety constraints (conservative)
        if level_measurement < self.level_limits[0] + self.level_margin:
            # Near lower limit - reduce base flow cautiously
            output = min(output, self.prev_output)
            if level_measurement < self.level_limits[0] + self.level_margin/2:
                output = min(output, self.prev_output - 0.1)
        elif level_measurement > self.level_limits[1] - self.level_margin:
            # Near upper limit - increase base flow cautiously
            output = max(output, self.prev_output)
            if level_measurement > self.level_limits[1] - self.level_margin/2:
                output = max(output, self.prev_output + 0.1)
        
        # Apply pH safety constraints (very conservative)
        if self.filtered_pH < self.pH_limits[0] + 1.0:  # Very conservative lower limit
            # Dangerously low pH - increase base flow to raise pH
            output = max(output, self.prev_output + 0.5)
            if self.filtered_pH < self.pH_limits[0] + 0.5:
                output = max(output, self.actuator_limits[1] * 0.8)  # Use high flow
        elif self.filtered_pH > self.pH_limits[1] - 1.0:  # Very conservative upper limit
            # Dangerously high pH - decrease base flow to lower pH
            output = min(output, self.prev_output - 0.5)
            if self.filtered_pH > self.pH_limits[1] - 0.5:
                output = min(output, self.actuator_limits[0])  # Use minimum flow
        
        # Rate limiting to prevent chattering (very conservative)
        max_change = self.max_rate
        if output > self.prev_output + max_change:
            output = self.prev_output + max_change
        elif output < self.prev_output - max_change:
            output = self.prev_output - max_change
        
        # Apply actuator limits
        output = np.clip(output, self.actuator_limits[0], self.actuator_limits[1])
        
        # Store previous output for next step
        self.prev_output = output
        
        # Store previous error
        self.prev_error = error
        
        return np.array([output])
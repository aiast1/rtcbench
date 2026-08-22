import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # Much more conservative PID tuning to avoid actuator chattering
        # The duty limit of 0.0041 means we can only move ~0.2 mL/s per 5s step
        self.Kp = 0.8      # reduced proportional gain
        self.Ki = 0.05     # very slow integral to avoid windup and chattering
        self.Kd = 0.2      # minimal derivative on noisy signal
        
        # State tracking
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_pH = 7.0
        self.pump_output = 14.22  # start at steady state
        
        # Anti-windup limits - tighter to prevent saturation cycling
        self.integral_max = 3.0
        self.integral_min = -3.0
        
        # Actuator limits
        self.pump_min = 0.0
        self.pump_max = 30.0
        
        # Duty limit: max change per step is sqrt(0.0041) ≈ 0.064 mL/s
        # But we need margin, so use 0.04 mL/s per step
        self.max_pump_change = 0.04
        
        # Level safety
        self.level_min = 5.0
        self.level_max = 30.0
        self.level_setpoint = 17.5  # target middle of range
        
        # pH safety
        self.pH_min = 4.0
        self.pH_max = 10.5
        
        # Measurement indices
        self.pH_idx = 0
        self.level_idx = 1
        
        # Derivative filter to reduce noise sensitivity
        self.derivative_filter = 0.2
        self.filtered_derivative = 0.0
        
        # Track setpoint changes for bumpless transfer
        self.last_setpoint = 6.5
        
        # Slow ramp tracking for large setpoint changes
        self.setpoint_ramp_rate = 0.1  # pH units per step
        self.target_setpoint = 6.5
        
    def reset(self):
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_pH = 7.0
        self.pump_output = 14.22
        self.filtered_derivative = 0.0
        self.last_setpoint = 6.5
        self.target_setpoint = 6.5
    
    def step(self, t, y, r, quality):
        # Extract measurements
        pH_measured = y[self.pH_idx]
        level_measured = y[self.level_idx]
        
        # Get setpoint (pH is the scored channel)
        new_setpoint = r[0] if not np.isnan(r[0]) else self.last_setpoint
        
        # Ramp setpoint changes slowly to avoid aggressive pump moves
        setpoint_change = new_setpoint - self.target_setpoint
        if abs(setpoint_change) > self.setpoint_ramp_rate:
            self.target_setpoint += np.sign(setpoint_change) * self.setpoint_ramp_rate
        else:
            self.target_setpoint = new_setpoint
        
        self.last_setpoint = new_setpoint
        setpoint = self.target_setpoint
        
        # Only use pH measurement if quality flag is good
        if quality[self.pH_idx]:
            pH = pH_measured
        else:
            pH = self.last_pH
        
        self.last_pH = pH
        
        # Compute error
        error = setpoint - pH
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        # Only integrate if not saturated or if error is reducing saturation
        if not (self.pump_output >= self.pump_max and error > 0) and \
           not (self.pump_output <= self.pump_min and error < 0):
            self.integral_error += error * self.dt
        
        # Clamp integral to prevent windup
        self.integral_error = np.clip(self.integral_error, 
                                       self.integral_min, 
                                       self.integral_max)
        I = self.Ki * self.integral_error
        
        # Derivative term with low-pass filtering to reduce noise
        raw_derivative = (error - self.last_error) / self.dt if self.dt > 0 else 0.0
        self.filtered_derivative = (self.derivative_filter * raw_derivative + 
                                    (1.0 - self.derivative_filter) * self.filtered_derivative)
        D = self.Kd * self.filtered_derivative
        
        self.last_error = error
        
        # Compute base control output
        u = P + I + D
        
        # Apply strict rate limiting to stay within duty cycle
        pump_command = self.pump_output + np.clip(u, -self.max_pump_change, self.max_pump_change)
        
        # Level-based safety constraint - gentle corrections
        if quality[self.level_idx]:
            level_error = self.level_setpoint - level_measured
            # Gentle level correction: -0.5 to +0.5 mL/s adjustment
            level_correction = np.clip(level_error * 0.02, -0.5, 0.5)
            pump_command += level_correction
            
            # Hard limits if level is critical
            if level_measured > self.level_max - 0.5:
                pump_command = min(pump_command, 8.0)
            elif level_measured < self.level_min + 0.5:
                pump_command = max(pump_command, 20.0)
        
        # Final saturation
        pump_command = np.clip(pump_command, self.pump_min, self.pump_max)
        
        # Ensure we don't exceed duty limit even with all corrections
        actual_change = pump_command - self.pump_output
        if abs(actual_change) > self.max_pump_change:
            pump_command = self.pump_output + np.sign(actual_change) * self.max_pump_change
        
        # Update state for next iteration
        self.pump_output = pump_command
        
        return np.array([pump_command])
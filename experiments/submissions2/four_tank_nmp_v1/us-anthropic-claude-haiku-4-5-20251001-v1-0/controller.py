import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # Balanced PID gains: increase from conservative baseline while maintaining safety
        self.Kp = np.array([1.2, 1.2])      # Increased proportional gain
        self.Ki = np.array([0.15, 0.15])    # Increased integral gain
        self.Kd = np.array([0.0, 0.0])      # No derivative on noisy measurements
        
        # Integral state and anti-windup
        self.integral_error = np.array([0.0, 0.0])
        self.integral_limit = 4.0
        
        # Actuator limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Last control output for slew rate limiting
        self.u_last = np.array([3.0, 3.0])
        
        # Moderate slew rate limit - balance between response and duty cycle
        self.max_slew = 1.0  # V per 2s period
        
        # Low-pass filter for noisy measurements
        self.y_filtered = np.array([0.0, 0.0])
        self.filter_alpha = 0.25  # Moderate filtering
        
        # Duty cycle tracking for safety
        self.duty_accumulator = 0.0
        self.duty_limit = 0.0037
        self.duty_safety_margin = 0.0030  # Slightly relaxed threshold
        
        # Track previous error for rate limiting
        self.prev_error = np.array([0.0, 0.0])
        
        # Adaptive gain reduction based on duty cycle
        self.gain_scale = 1.0
        
    def reset(self):
        """Reset controller state for new scenario."""
        self.integral_error = np.array([0.0, 0.0])
        self.u_last = np.array([3.0, 3.0])
        self.y_filtered = np.array([0.0, 0.0])
        self.duty_accumulator = 0.0
        self.prev_error = np.array([0.0, 0.0])
        self.gain_scale = 1.0
    
    def step(self, t, y, r, quality):
        """
        Balanced PID controller: improved tracking while maintaining safety.
        
        Strategy:
        - Moderate filtering to reduce noise impact
        - Increased gains for better tracking
        - Adaptive gain reduction only when duty cycle is critical
        - Smooth error rate limiting to avoid jerky responses
        """
        
        # Use only good quality measurements
        y_valid = np.where(quality, y, self.y_filtered)
        
        # Moderate low-pass filter
        self.y_filtered = self.filter_alpha * y_valid + (1.0 - self.filter_alpha) * self.y_filtered
        
        # Compute tracking error
        error = r - self.y_filtered
        
        # Handle NaN setpoints (non-scored channels)
        error = np.where(np.isnan(error), 0.0, error)
        
        # Gentle error rate limiting to smooth transients
        error_rate = (error - self.prev_error) / self.dt
        error_rate_limit = 1.0  # cm/s (relaxed from 0.5)
        error_rate = np.clip(error_rate, -error_rate_limit, error_rate_limit)
        error = self.prev_error + error_rate * self.dt
        self.prev_error = error.copy()
        
        # Proportional term with adaptive gain
        p_term = self.Kp * self.gain_scale * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        
        # Clamp integral state
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit
        )
        
        i_term = self.Ki * self.gain_scale * self.integral_error
        
        # Compute raw control signal
        u_raw = p_term + i_term
        
        # Apply actuator hard limits
        u_limited = np.clip(u_raw, self.u_min, self.u_max)
        
        # Apply slew rate limiting
        u_slew = np.clip(
            u_limited,
            self.u_last - self.max_slew,
            self.u_last + self.max_slew
        )
        
        # Anti-windup: if we saturated, decay integral error
        saturated = (u_limited != u_raw)
        self.integral_error[saturated] *= 0.90
        
        # Track duty cycle
        duty_increment = np.sum(np.abs(u_slew - self.u_last)) / (2.0 * 10.0)
        self.duty_accumulator += duty_increment
        
        # Adaptive gain reduction based on duty cycle
        if self.duty_accumulator > self.duty_safety_margin:
            # Gradually reduce gain as we approach limit
            excess = (self.duty_accumulator - self.duty_safety_margin) / (self.duty_limit - self.duty_safety_margin)
            self.gain_scale = max(0.5, 1.0 - 0.5 * excess)
            self.integral_error *= 0.92
        else:
            # Gradually restore gain when duty cycle is low
            self.gain_scale = min(1.0, self.gain_scale + 0.02)
        
        # Reduce chattering near setpoint
        small_error_threshold = 0.2  # cm
        if np.all(np.abs(error) < small_error_threshold):
            u_slew = 0.9 * u_slew + 0.1 * self.u_last
        
        # Update last control output
        self.u_last = u_slew.copy()
        
        return u_slew
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        
        # Very conservative PID parameters - focus on stability
        self.Kp = 4.0      # Further reduced
        self.Ki = 0.15     # Further reduced
        self.Kd = 6.0      # Further reduced
        
        # Heavy filtering
        self.derivative_filter_tau = 25.0
        self.measurement_filter_tau = 15.0
        
        # Anti-windup
        self.integral_limit = 15.0
        
        # Actuator constraints
        self.actuator_min = 270.0
        self.actuator_max = 340.0
        self.actuator_slew_limit = 1.5 * self.sample_time
        
        # Conservative safety margins
        self.temp_lower_safety = 330.0  # Much higher margin
        self.temp_upper_safety = 430.0  # Much lower margin
        
        # Emergency thresholds
        self.temp_emergency_low = 310.0
        self.temp_emergency_high = 465.0
        
        # State variables
        self.prev_measurement: Optional[float] = None
        self.prev_filtered_measurement: Optional[float] = None
        self.prev_filtered_derivative: Optional[float] = None
        self.prev_output: Optional[float] = None
        self.integral: float = 0.0
        self.prev_error: Optional[float] = None
        
        # Bumpless transfer
        self.initial_output = 300.0
        
        # Safety state
        self.safety_override_active = False
        self.safety_override_counter = 0
        
        # Derivative spike detection
        self.max_derivative_allowed = 3.0
        
        # Error deadband to reduce chattering
        self.error_deadband = 0.5
        
        # Output smoothing
        self.output_filter_tau = 5.0
        self.prev_filtered_output: Optional[float] = None
        
    def reset(self):
        """Reset controller state for new scenario"""
        self.prev_measurement = None
        self.prev_filtered_measurement = None
        self.prev_filtered_derivative = None
        self.prev_output = self.initial_output
        self.prev_filtered_output = self.initial_output
        self.integral = 0.0
        self.prev_error = None
        self.safety_override_active = False
        self.safety_override_counter = 0
        
    def step(self, t, y, r, quality):
        """
        Ultra-conservative control with maximum safety focus
        """
        # Extract current measurement and setpoint
        current_measurement = y[0]
        setpoint = r[0]
        
        # Handle bad quality measurement
        if not quality[0]:
            if self.prev_measurement is not None:
                current_measurement = self.prev_measurement
            else:
                if self.prev_output is not None:
                    return np.array([self.prev_output])
                else:
                    return np.array([self.initial_output])
        
        # Initialize on first step
        if self.prev_measurement is None:
            self.prev_measurement = current_measurement
            self.prev_filtered_measurement = current_measurement
            self.prev_filtered_derivative = 0.0
            self.prev_filtered_output = self.initial_output
            return np.array([self.initial_output])
        
        # Apply heavy measurement filtering
        alpha_meas = self.sample_time / (self.measurement_filter_tau + self.sample_time)
        filtered_measurement = (alpha_meas * current_measurement + 
                               (1 - alpha_meas) * self.prev_filtered_measurement)
        
        # Check for emergency conditions IMMEDIATELY
        if filtered_measurement > self.temp_emergency_high:
            # Emergency cooling - maximum cooling
            self.safety_override_active = True
            self.safety_override_counter = 10  # Hold for 10 samples
            raw_output = self.actuator_min
            self.integral = 0.0  # Reset integral
        elif filtered_measurement < self.temp_emergency_low:
            # Emergency heating - maximum heating
            self.safety_override_active = True
            self.safety_override_counter = 10  # Hold for 10 samples
            raw_output = self.actuator_max
            self.integral = 0.0  # Reset integral
        elif self.safety_override_active:
            # Count down safety override
            self.safety_override_counter -= 1
            if self.safety_override_counter <= 0:
                self.safety_override_active = False
            # During override, maintain conservative action
            if filtered_measurement > self.temp_upper_safety:
                raw_output = self.actuator_min
            elif filtered_measurement < self.temp_lower_safety:
                raw_output = self.actuator_max
            else:
                # Gradual return to normal
                raw_output = 295.0  # Bias toward cooling
        else:
            # Normal control with enhanced safety checks
            
            # Calculate error with deadband
            error = setpoint - filtered_measurement
            if abs(error) < self.error_deadband:
                error = 0.0
            
            # Apply error limiting for safety
            error_limited = np.clip(error, -15.0, 15.0)
            
            # Proportional term
            proportional = self.Kp * error_limited
            
            # Integral term - VERY conservative
            # Only integrate when close to setpoint and within safe zone
            safe_zone = self.temp_lower_safety < filtered_measurement < self.temp_upper_safety
            small_error = abs(error) < 5.0
            
            if safe_zone and small_error:
                self.integral += self.Ki * error * self.sample_time
            else:
                # Freeze or reduce integral
                self.integral *= 0.95  # Leaky integrator
            
            # Clamp integral tightly
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
            
            # Derivative term - heavily filtered and limited
            derivative = (filtered_measurement - self.prev_filtered_measurement) / self.sample_time
            
            # Hard limit on derivative
            derivative = np.clip(derivative, -self.max_derivative_allowed, self.max_derivative_allowed)
            
            # Apply heavy derivative filtering
            alpha_deriv = self.sample_time / (self.derivative_filter_tau + self.sample_time)
            filtered_derivative = (alpha_deriv * derivative + 
                                  (1 - alpha_deriv) * self.prev_filtered_derivative)
            
            derivative_term = -self.Kd * filtered_derivative
            
            # NO feedforward - too risky
            
            # Calculate raw PID output
            pid_output = proportional + self.integral + derivative_term
            
            # Base output biased toward cooling for unstable reactor
            base_output = 290.0  # Even cooler bias
            
            raw_output = base_output + pid_output
            
            # Apply proactive safety constraints
            if filtered_measurement < self.temp_lower_safety:
                # Getting cold - reduce cooling aggressively
                raw_output = min(raw_output, 325.0)
                self.integral = max(self.integral, 0)  # Prevent negative windup
            elif filtered_measurement > self.temp_upper_safety:
                # Getting hot - increase cooling aggressively
                raw_output = max(raw_output, 275.0)
                self.integral = min(self.integral, 0)  # Prevent positive windup
            
            # Additional safety: if temperature is changing too fast, take action
            if abs(filtered_derivative) > 2.0:
                if filtered_derivative > 0:  # Heating up fast
                    raw_output = max(raw_output, 280.0)
                else:  # Cooling down fast
                    raw_output = min(raw_output, 320.0)
        
        # Apply actuator limits
        constrained_output = np.clip(raw_output, self.actuator_min, self.actuator_max)
        
        # Apply conservative slew rate limiting
        if self.prev_output is not None:
            # Very conservative slew limiting
            max_change = 0.5 * self.actuator_slew_limit
            
            slew_limited_output = np.clip(
                constrained_output,
                self.prev_output - max_change,
                self.prev_output + max_change
            )
            
            # Additional output filtering for smoothness
            if self.prev_filtered_output is not None:
                alpha_out = self.sample_time / (self.output_filter_tau + self.sample_time)
                filtered_output = (alpha_out * slew_limited_output + 
                                 (1 - alpha_out) * self.prev_filtered_output)
            else:
                filtered_output = slew_limited_output
        else:
            filtered_output = constrained_output
        
        # Final actuator value
        final_output = filtered_output
        
        # Update state variables
        self.prev_measurement = current_measurement
        self.prev_filtered_measurement = filtered_measurement
        self.prev_filtered_derivative = filtered_derivative
        self.prev_output = final_output
        self.prev_filtered_output = filtered_output if 'filtered_output' in locals() else final_output
        
        return np.array([final_output])
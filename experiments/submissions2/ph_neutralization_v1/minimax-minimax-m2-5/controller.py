import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_min = 0.0
        self.actuator_max = 30.0
        self.initial_actuator = 14.22
        
        # PID gains (tuned for slow dynamics and delay)
        self.Kp = 0.08
        self.Ki = 0.005
        
        # Anti-windup back-calculation gain
        self.Kt = 2.0
        
        # Measurement filter (low-pass)
        self.filter_alpha = 0.3
        
        # Actuator rate limit (mL/s per step)
        self.max_delta_u = 0.1
        
        # Level constraints and nominal operating point
        self.level_min = 5.0
        self.level_max = 30.0
        self.level_nom = 17.5
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        # Integrator state (set to initial actuator for bumpless start)
        self.integrator = self.initial_actuator
        
        # Previous error and measurement
        self.prev_error = 0.0
        self.filtered_measurement = 6.5  # assume starts at initial setpoint
        self.prev_measurement = 6.5
        
        # Previous actuator value
        self.prev_actuator = self.initial_actuator
        
        # Level measurement (initialize to nominal)
        self.level_meas = self.level_nom
    
    def step(self, t, y, r, quality):
        # Extract measurements
        pH_meas = y[0]
        level_meas = y[1]
        
        # Update level measurement
        self.level_meas = level_meas
        
        # Handle measurement quality: use filtered value if bad
        if quality[0]:
            # Update low-pass filter
            self.filtered_measurement = self.filter_alpha * pH_meas + (1 - self.filter_alpha) * self.filtered_measurement
        # else: retain previous filtered_measurement
        
        # Get setpoint (only pH is scored)
        pH_setpoint = r[0]
        
        # Compute error
        error = pH_setpoint - self.filtered_measurement
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup (back-calculation)
        # Compute unsaturated control signal
        u_unsat = P + self.integrator
        
        # Check actuator limits and apply anti-windup
        if u_unsat > self.actuator_max:
            u_sat = self.actuator_max
            # Back-calculation: adjust integrator based on saturation
            self.integrator += self.Ki * (u_sat - u_unsat) * self.sample_time
        elif u_unsat < self.actuator_min:
            u_sat = self.actuator_min
            self.integrator += self.Ki * (u_sat - u_unsat) * self.sample_time
        else:
            u_sat = u_unsat
            # Normal integration
            self.integrator += self.Ki * error * self.sample_time
        
        # Apply rate limiting to prevent actuator chattering
        delta_u = u_sat - self.prev_actuator
        if delta_u > self.max_delta_u:
            u_sat = self.prev_actuator + self.max_delta_u
        elif delta_u < -self.max_delta_u:
            u_sat = self.prev_actuator - self.max_delta_u
        
        # Level supervision: adjust actuator to keep level within bounds
        # Use a simple P controller to nudge actuator away from limits
        if self.level_meas > self.level_max - 2.0:
            # Level too high, reduce actuator
            u_sat -= 0.05 * (self.level_meas - (self.level_max - 2.0))
        elif self.level_meas < self.level_min + 2.0:
            # Level too low, increase actuator
            u_sat += 0.05 * ((self.level_min + 2.0) - self.level_meas)
        
        # Final actuator clamp
        u = np.clip(u_sat, self.actuator_min, self.actuator_max)
        
        # Update state for next step
        self.prev_error = error
        self.prev_measurement = self.filtered_measurement
        self.prev_actuator = u
        
        return np.array([u])
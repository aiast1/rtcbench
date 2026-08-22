import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Actuator limits
        self.u_min = 0.0
        self.u_max = 30.0
        
        # Safety limits - pH must stay in [4.0, 10.5]
        self.pH_min = 4.0
        self.pH_max = 10.5
        self.level_min = 5.0
        self.level_max = 30.0
        
        # Very conservative gains for robustness with 20s delay
        self.Kc = 0.08  # Proportional gain - very conservative
        self.Ti = 300.0  # Integral time (s) - very slow
        self.Td = 0.0  # No derivative
        
        # State variables
        self.integral = 0.0
        self.prev_u = 14.22
        self.u_filtered = 14.22
        
        # Output filter
        self.alpha_u = 0.25
        
        # Reference filter
        self.sp_filtered = 6.5
        
        # Quality tracking
        self.bad_count = 0
        self.last_good_pH = 6.5
        self.last_good_level = 17.5
        
        # Safety margins - keep pH well within bounds
        self.pH_low_limit = 4.5  # Safety margin above 4.0
        self.pH_high_limit = 10.0  # Safety margin below 10.5
        
    def reset(self):
        self.integral = 0.0
        self.prev_u = 14.22
        self.u_filtered = 14.22
        self.sp_filtered = 6.5
        self.bad_count = 0
        self.last_good_pH = 6.5
        self.last_good_level = 17.5
        
    def step(self, t, y, r, quality):
        # Extract measurements
        pH_meas = y[0]
        level_meas = y[1]
        pH_quality = quality[0]
        level_quality = quality[1]
        
        # Handle bad quality readings
        if pH_quality:
            self.last_good_pH = pH_meas
            self.bad_count = 0
        else:
            self.bad_count += 1
            pH_meas = self.last_good_pH if self.bad_count < 10 else r[0]
            
        if level_quality:
            self.last_good_level = level_meas
        else:
            level_meas = self.last_good_level
            
        # Get setpoint
        sp = r[0]
        
        # Constrain setpoint to safe region
        sp_safe = np.clip(sp, self.pH_low_limit, self.pH_high_limit)
        
        # Smooth setpoint changes - very slow ramping
        sp_rate = 0.008  # Very slow setpoint tracking
        if sp_safe > self.sp_filtered + sp_rate:
            self.sp_filtered += sp_rate
        elif sp_safe < self.sp_filtered - sp_rate:
            self.sp_filtered -= sp_rate
        else:
            self.sp_filtered = sp_safe
            
        # Calculate error
        error = self.sp_filtered - pH_meas
        
        # Proportional term
        P = self.Kc * error
        
        # Integral term with strong anti-windup
        dt = self.sample_time
        
        # Only integrate if we're in safe region
        integrate = True
        if pH_meas < self.pH_low_limit and error < 0:
            # pH too low and trying to go lower - stop integrating negative
            integrate = False
        elif pH_meas > self.pH_high_limit and error > 0:
            # pH too high and trying to go higher - stop integrating positive
            integrate = False
            
        if integrate:
            delta_I = self.Kc * dt / self.Ti * error
            self.integral += delta_I
            
        # Clamp integral
        max_integral = 10.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        I = self.integral
        
        # Calculate raw output
        u_raw = P + I
        
        # Safety override: if pH is in danger zone, take strong corrective action
        if pH_meas < self.pH_low_limit:
            # pH too low - add base aggressively
            safety_boost = 3.0 * (self.pH_low_limit - pH_meas)
            u_raw += safety_boost
        elif pH_meas > self.pH_high_limit:
            # pH too high - reduce base aggressively
            safety_cut = 3.0 * (pH_meas - self.pH_high_limit)
            u_raw -= safety_cut
            
        # Level-based constraints
        if level_meas < self.level_min + 5.0:
            # Level low - reduce base flow proportionally
            level_factor = max(0.2, (level_meas - self.level_min) / 5.0)
            u_raw = 14.22 + (u_raw - 14.22) * level_factor
            
        # Clamp to actuator limits
        u_clamped = np.clip(u_raw, self.u_min, self.u_max)
        
        # Rate limit the output - very conservative
        max_slew = 1.0
        if self.prev_u is not None:
            u_clamped = np.clip(u_clamped, 
                               self.prev_u - max_slew, 
                               self.prev_u + max_slew)
            
        # Filter output
        self.u_filtered = self.alpha_u * u_clamped + (1 - self.alpha_u) * self.u_filtered
        u_final = self.u_filtered
        
        # Final clamp
        u_final = np.clip(u_final, self.u_min, self.u_max)
        
        self.prev_u = u_final
        
        return np.array([u_final])
import numpy as np

class Controller:
    """
    Controller for unstable CSTR operating at middle steady state.
    Uses PI control with derivative filtering and anti-windup.
    """
    
    def __init__(self, brief):
        self.Ts = brief.sample_time
        
        # PI controller parameters - tuned for unstable CSTR
        self.Kc = 2.5  # Proportional gain
        self.tau_i = 80.0  # Integral time (s)
        self.tau_d = 12.0  # Derivative time (s)
        
        # Anti-windup limits
        self.Tj_min = 270.0
        self.Tj_max = 340.0
        
        # Safety limits for reactor temperature
        self.T_safe_min = 302.0
        self.T_safe_max = 468.0
        
        # Nominal operating point
        self.T_op = 350.0
        self.Tj_op = 300.0
        
        # Filter coefficient for derivative
        self.N = 8.0
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_T = None
        self.deriv_state = 0.0
        self.Tj_prev = self.Tj_op
        self.first_step = True
        
        # Quality tracking
        self.last_good_T = self.T_op
        self.last_good_Tj = self.Tj_op
    
    def _rate_limit(self, u, u_prev, max_rate):
        """Apply rate limiting."""
        if u_prev is None:
            return u
        du = u - u_prev
        max_du = max_rate * self.Ts
        return u_prev + np.clip(du, -max_du, max_du)
    
    def step(self, t, y, r, quality):
        """Execute one control step."""
        # Extract measurements
        T_meas = y[0]  # Reactor temperature
        Tj_meas = y[1]  # Jacket temperature
        
        # Extract setpoint
        T_set = r[0] if not np.isnan(r[0]) else self.T_op
        
        # Handle bad quality readings
        T_good = quality[0] if len(quality) > 0 else True
        Tj_good = quality[1] if len(quality) > 1 else True
        
        # Update last good readings
        if T_good:
            self.last_good_T = T_meas
        if Tj_good:
            self.last_good_Tj = Tj_meas
        
        # Use last good values if current is bad
        T = self.last_good_T if not T_good else T_meas
        Tj = self.last_good_Tj if not Tj_good else Tj_meas
        
        # Safety override - if approaching limits, take aggressive action
        if T > self.T_safe_max - 10:
            # Emergency cooling
            Tj_set = self.Tj_min
            self.integral = max(self.integral, -30.0)
        elif T < self.T_safe_min + 3:
            # Emergency heating
            Tj_set = self.Tj_max
            self.integral = min(self.integral, 30.0)
        else:
            # Normal control operation
            
            # Compute error
            error = T_set - T
            
            # Proportional term
            P = self.Kc * error
            
            # Integral term with anti-windup
            # Reduce integral gain when far from setpoint
            integral_gain_factor = 1.0
            if abs(error) > 8:
                integral_gain_factor = 0.3
            elif abs(error) > 4:
                integral_gain_factor = 0.6
            
            self.integral += integral_gain_factor * self.Kc / self.tau_i * error * self.Ts
            
            # Anti-windup: clamp integral
            max_integral = 25.0
            self.integral = np.clip(self.integral, -max_integral, max_integral)
            
            # Derivative term (on measurement to avoid derivative kick)
            if self.prev_T is not None:
                dT = T - self.prev_T
                # Filtered derivative
                self.deriv_state = (self.tau_d * dT / self.Ts + self.deriv_state * self.N * self.tau_d) / (1 + self.N * self.tau_d)
                D = -self.Kc * self.deriv_state
            else:
                D = 0.0
            
            # Compute jacket setpoint
            Tj_set = self.Tj_op + P + self.integral + D
        
        # Rate limit the output (1.5 K/s max slew)
        max_slew = 1.4  # Slightly conservative
        Tj_set = self._rate_limit(Tj_set, self.Tj_prev, max_slew)
        
        # Hard limits
        Tj_set = np.clip(Tj_set, self.Tj_min, self.Tj_max)
        
        # Handle first step - bumpless transfer
        if self.first_step:
            Tj_set = self.Tj_op
            self.first_step = False
        
        # Update state
        self.prev_T = T
        self.Tj_prev = Tj_set
        
        return np.array([Tj_set])
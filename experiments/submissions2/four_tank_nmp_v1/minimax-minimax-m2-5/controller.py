import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = brief.sample_time
        
        # Very conservative PI gains for NMP system with unmeasured states
        self.Kp = np.array([0.15, 0.15])
        self.Ki = np.array([0.008, 0.008])
        
        # Anti-windup
        self.Kt = np.array([0.1, 0.1])
        
        # Actuator limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        self.u_bumpless = np.array([3.0, 3.0])
        
        # Nominal parameters for state estimation
        self.A1 = 28.0
        self.A2 = 32.0
        self.A3 = 28.0
        self.A4 = 32.0
        self.a1 = 0.071
        self.a2 = 0.057
        self.a3 = 0.071
        self.a4 = 0.057
        self.k1 = 3.14
        self.k2 = 3.29
        self.gamma1 = 0.43
        self.gamma2 = 0.34
        
        # State
        self.integrator = np.zeros(2)
        self.prev_u = None
        self.prev_y = None
        
        # Estimate of unmeasured upper tank levels
        self.h3_est = 10.0
        self.h4_est = 10.0
        
        # Safety margins
        self.max_safe_lower = 14.0  # Leave headroom for upper tanks
        self.max_u_step = 0.5  # Limit actuator changes for smoothness
        
    def reset(self):
        """Reset controller state for new scenario"""
        self.integrator = np.zeros(2)
        self.prev_u = self.u_bumpless.copy()
        self.prev_y = None
        # Reasonable initial estimates for upper tanks
        self.h3_est = 10.0
        self.h4_est = 10.0
        
    def estimate_upper_tanks(self, y, u, dt):
        """Estimate unmeasured upper tank levels using simplified dynamics"""
        # Inflow to upper tanks
        # Pump 1: (1-gamma1) goes to tank 4, pump 2: (1-gamma2) goes to tank 3
        q1_in = self.k1 * u[0] * (1.0 - self.gamma1)
        q2_in = self.k2 * u[1] * (1.0 - self.gamma2)
        
        # Outflow from upper tanks (drain into lower tanks)
        q34_out = self.a3 * np.sqrt(max(self.h3_est, 0)) + self.a4 * np.sqrt(max(self.h4_est, 0))
        
        # Simple split based on relative levels
        if self.h3_est + self.h4_est > 0:
            frac3 = self.h3_est / (self.h3_est + self.h4_est)
            frac4 = self.h4_est / (self.h3_est + self.h4_est)
        else:
            frac3 = 0.5
            frac4 = 0.5
            
        q3_out = q34_out * frac3
        q4_out = q34_out * frac4
        
        # Update estimates
        self.h3_est += (q2_in - q3_out) * dt / self.A3
        self.h4_est += (q1_in - q4_out) * dt / self.A4
        
        # Clamp to physical limits
        self.h3_est = np.clip(self.h3_est, 0, 20)
        self.h4_est = np.clip(self.h4_est, 0, 20)
        
    def step(self, t, y, r, quality):
        """
        t: time in seconds
        y: measured outputs [tank1, tank2]
        r: setpoints [tank1_setpoint, tank2_setpoint]
        quality: boolean array indicating measurement validity
        """
        # Handle NaN setpoints
        r = np.where(np.isnan(r), y, r)
        
        # Default quality
        if quality is None:
            quality = np.array([True, True])
            
        # Initialize on first call
        if self.prev_y is None:
            self.prev_y = y.copy()
            # Estimate initial upper tank levels based on steady state
            # At u=[3,3], lower tanks are at setpoint, upper tanks at some equilibrium
            self.h3_est = 8.0  # Reasonable guess
            self.h4_est = 8.0
            return self.u_bumpless.copy()
        
        # Estimate upper tank states
        self.estimate_upper_tanks(y, self.prev_u, self.Ts)
        
        # Compute error
        error = r - y
        
        # Check if upper tanks are getting too high - constrain effective setpoint
        # If upper tanks > 15cm, reduce lower tank setpoint demand
        safety_factor = 1.0
        if self.h3_est > 15.0:
            safety_factor = min(safety_factor, 0.5)
        if self.h4_est > 15.0:
            safety_factor = min(safety_factor, 0.5)
            
        # Apply safety factor to error (reduces integral buildup when upper tanks high)
        error_safe = error * safety_factor
        
        # PI control with anti-windup
        for i in range(2):
            if quality[i]:
                # Compute desired integral update
                i_term_new = self.integrator[i] + self.Ki[i] * error_safe[i]
                
                # Compute total control
                u_desired = self.Kp[i] * error_safe[i] + i_term_new
                
                # Apply saturation
                u_sat = np.clip(u_desired, self.u_min[i], self.u_max[i])
                
                # Back-calculation anti-windup
                if u_desired != u_sat:
                    self.integrator[i] += self.Ki[i] * (u_sat - u_desired) * self.Kt[i]
                else:
                    self.integrator[i] = i_term_new
                    
                # Clamp integrator to prevent windup
                self.integrator[i] = np.clip(self.integrator[i], 
                                             self.u_min[i] - self.Kp[i] * error_safe[i],
                                             self.u_max[i] - self.Kp[i] * error_safe[i])
        
        # Compute final control
        u = np.zeros(2)
        for i in range(2):
            if quality[i]:
                u[i] = self.Kp[i] * error_safe[i] + self.integrator[i]
            else:
                u[i] = self.prev_u[i]
        
        # Apply hard limits
        u = np.clip(u, self.u_min, self.u_max)
        
        # Rate limiting for smoothness and to prevent duty cycle issues
        if self.prev_u is not None:
            u_diff = u - self.prev_u
            # Limit the change rate
            max_change = self.max_u_step
            u_diff_limited = np.clip(u_diff, -max_change, max_change)
            u = self.prev_u + u_diff_limited
            
            # Re-apply saturation after rate limiting
            u = np.clip(u, self.u_min, self.u_max)
        
        # Store state
        self.prev_u = u.copy()
        self.prev_y = y.copy()
        
        return u
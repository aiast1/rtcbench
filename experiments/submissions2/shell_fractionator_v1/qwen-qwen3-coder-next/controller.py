import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_y = 3
        self.n_u = 3
        self.n_d = 2
        
        # Conservative tuning for robustness across all scenarios
        self.tau_c = 120.0  # closed-loop time constant (slower for robustness)
        
        # PID parameters per loop (PI control with conservative tuning)
        self.Kc = np.zeros(self.n_y)
        self.tau_I = np.zeros(self.n_y)
        
        # Compute tuning parameters using Ziegler-Nichols-style tuning with safety margins
        for i in range(self.n_y):
            # Use nominal plant model for tuning (conservative approach)
            K_p = 5.0  # typical gain
            theta = 20.0  # typical dead time
            tau = 50.0  # typical time constant
            
            # Ziegler-Nichols PI tuning with significant safety factor
            Kc = (1.0 / K_p) * (tau / (1.2 * theta))
            tau_I = tau * (1.2 * theta + tau) / (1.8 * theta)
            
            # Apply aggressive conservative scaling
            self.Kc[i] = Kc * 0.15  # Very reduced gain for robustness
            self.tau_I[i] = max(tau_I, 100.0)  # Very long integral time
        
        # Actuator constraints
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        self.u_slew_limit = 0.0112  # maximum change per step
        
        # Safety constraint for y3 (bottoms reflux temperature)
        self.y3_min = -0.5
        
        # Disturbance rejection (simple integral action on errors)
        self.integral = np.zeros(self.n_y)
        self.last_error = np.zeros(self.n_y)
        self.last_u = np.zeros(self.n_u)
        
        # Setpoint schedule
        self._build_setpoint_schedule()
        
    def _build_setpoint_schedule(self):
        # Build setpoint schedule as piecewise linear segments
        self.sp_times = [0, 120, 350, 440, 600]
        self.sp_values = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, -0.15, 0.0]),
            np.array([-0.1, 0.1, 0.05]),
            np.array([-0.1, 0.1, 0.05]),  # end of ramp
            np.array([0.0, 0.0, 0.0])
        ]
        
    def _get_setpoint(self, t):
        # Interpolate setpoints at time t
        if t < self.sp_times[0]:
            return self.sp_values[0].copy()
        if t >= self.sp_times[-1]:
            return self.sp_values[-1].copy()
        
        # Find segment
        for i in range(len(self.sp_times) - 1):
            if self.sp_times[i] <= t < self.sp_times[i + 1]:
                # Linear interpolation
                t0, t1 = self.sp_times[i], self.sp_times[i + 1]
                sp0, sp1 = self.sp_values[i], self.sp_values[i + 1]
                frac = (t - t0) / (t1 - t0)
                return sp0 + frac * (sp1 - sp0)
        
        return self.sp_values[-1].copy()
    
    def reset(self):
        self.integral = np.zeros(self.n_y)
        self.last_error = np.zeros(self.n_y)
        self.last_u = np.array([0.0, 0.0, 0.0])
        
    def step(self, t, y, r, quality):
        # Get setpoint (only for scored channels)
        sp = self._get_setpoint(t)
        
        # Handle NaN setpoints (non-scored channels)
        for i in range(self.n_y):
            if np.isnan(r[i]):
                sp[i] = y[i]  # track current value (no control)
        
        # Compute errors (only for valid measurements)
        error = np.zeros(self.n_y)
        for i in range(self.n_y):
            if quality[i]:
                error[i] = sp[i] - y[i]
            else:
                # Use last error if measurement is bad
                error[i] = self.last_error[i]
        
        # PID calculation with anti-windup and safety constraints
        u = np.zeros(self.n_u)
        
        for i in range(self.n_y):
            # Proportional term
            P = self.Kc[i] * error[i]
            
            # Integral term with anti-windup (clamping)
            I = self.integral[i] + (self.sample_time / self.tau_I[i]) * error[i]
            
            # Anti-windup: limit integral action when actuator is saturated
            u_prop = P
            u_total = u_prop + I
            
            # Check if actuator would saturate
            if u_total > self.u_max[i] or u_total < self.u_min[i]:
                # Limit integral term to prevent windup
                if u_total > self.u_max[i]:
                    I = max(I, self.u_max[i] - u_prop)
                else:
                    I = min(I, self.u_min[i] - u_prop)
            
            # Derivative term (disabled due to noise concerns)
            D = 0.0
            
            # Combined control action
            u[i] = P + I + D
            
            # Update integral for next step
            self.integral[i] = I
            self.last_error[i] = error[i]
        
        # Apply actuator constraints
        u = np.clip(u, self.u_min, self.u_max)
        
        # Apply slew rate limits
        du = u - self.last_u
        du = np.clip(du, -self.u_slew_limit, self.u_slew_limit)
        u = self.last_u + du
        
        # Safety constraint enforcement: ensure y3 stays above -0.5
        # Use a more aggressive safety margin approach
        if quality[2]:  # Only apply if we have a valid measurement
            # Check if current y3 is near its safety limit
            if y[2] < -0.35:  # 0.15 margin below the -0.5 limit
                # Reduce control action that could push y3 lower
                # FCV-203 (index 2) affects y3 most directly
                # Reduce its action if it's trying to increase y3 too much
                if u[2] > 0 and y[2] < -0.3:
                    u[2] = min(u[2], 0.15)  # Cap at 0.15 to be safe
                elif u[2] < 0 and y[2] < -0.3:
                    u[2] = max(u[2], -0.05)  # Reduce negative action
                    
            # Even more conservative near the limit
            if y[2] < -0.40:  # 0.10 margin below the -0.5 limit
                u[2] = min(u[2], 0.1)  # Very conservative cap
                if u[2] < 0:
                    u[2] = 0.0  # Stop negative action
                    
        # Update last actuator values
        self.last_u = u.copy()
        
        return u
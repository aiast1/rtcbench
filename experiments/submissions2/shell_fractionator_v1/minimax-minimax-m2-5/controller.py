import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Control parameters - tuned for robustness given plant uncertainties
        # Using conservative PI due to significant dead time and lag
        self.Kp = np.array([0.08, 0.06, 0.10])  # Proportional gains
        self.Ki = np.array([0.004, 0.003, 0.006])  # Integral gains
        self.Kd = np.array([0.0, 0.0, 0.0])  # No derivative - too noisy with delay
        
        # Anti-windup parameters
        self.Kt = 0.1  # Back-calculation gain for anti-windup
        
        # Output limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint
        self.y2_min = -0.5
        
        # State variables
        self.integrator = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_y = np.zeros(3)
        self.u_prev = np.zeros(3)
        
        # Filter for derivative (not used but prepared)
        self.alpha = 0.1
        
    def reset(self):
        """Reset controller state for new scenario"""
        self.integrator = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_y = np.zeros(3)
        self.u_prev = np.zeros(3)
        
    def compute_setpoint(self, t):
        """Compute setpoints based on schedule"""
        r = np.zeros(3)
        
        if t < 120:
            r = np.array([0.0, 0.0, 0.0])
        elif t < 350:
            r = np.array([0.2, -0.15, 0.0])
        elif t < 440:
            # Ramp from [0.2, -0.15, 0.0] to [-0.1, 0.1, 0.05] over 90s
            frac = (t - 350) / 90.0
            r = np.array([0.2 - 0.3*frac, -0.15 + 0.25*frac, 0.0 + 0.05*frac])
        elif t < 600:
            r = np.array([-0.1, 0.1, 0.05])
        else:
            r = np.array([0.0, 0.0, 0.0])
            
        return r
    
    def step(self, t, y, r, quality):
        """
        Main control step
        
        Args:
            t: current time in seconds
            y: measured outputs (3,)
            r: setpoints (3,) - nan for unscored channels
            quality: bool array (3,) - False means stale/bad reading
            
        Returns:
            u: control outputs (3,)
        """
        # Use provided setpoints if available, otherwise compute from schedule
        if np.any(~np.isnan(r)):
            setpoint = r
        else:
            setpoint = self.compute_setpoint(t)
        
        # Handle bad quality measurements with hold-last-value
        y_filtered = np.where(quality, y, self.prev_y)
        
        # Compute error
        error = setpoint - y_filtered
        
        # Safety constraint: ensure y[2] stays above -0.5
        # If y[2] is close to limit, add bias to setpoint to push it up
        if y_filtered[2] < -0.4:
            error[2] -= 0.1  # Additional correction
        
        # Compute proportional term
        P = self.Kp * error
        
        # Compute integral term with anti-windup
        # First, compute the unconstrained control signal
        u_unsat = P + self.integrator
        
        # Saturate
        u = np.clip(u_unsat, self.u_min, self.u_max)
        
        # Anti-windup: back-calculation
        u_sat = u - u_unsat  # Saturation error
        self.integrator += self.Ki * error * self.sample_time + self.Kt * u_sat
        
        # Clamp integrator to prevent windup beyond reasonable bounds
        self.integrator = np.clip(self.integrator, self.u_min - self.Kp * error, 
                                   self.u_max - self.Kp * error)
        
        # Additional safety: hard constraint on y[2]
        # If measurement is bad or we're too close to limit, reduce u[2]
        if not quality[2]:
            # Hold last good output
            u[2] = self.u_prev[2]
        elif y_filtered[2] < -0.45:
            # Emergency: push temperature up
            u[2] = min(u[2], -0.1)  # Allow more cooling (negative = more duty)
        
        # Rate limiting to prevent actuator chatter
        max_rate = 0.0112  # Duty limit from problem
        du = u - self.u_prev
        rate = np.abs(du) / self.sample_time
        
        # If rate exceeds limit, scale back
        if np.any(rate > max_rate):
            scale = max_rate / np.max(rate)
            u = self.u_prev + du * scale
        
        # Store for next iteration
        self.prev_error = error
        self.prev_y = y_filtered
        self.u_prev = u.copy()
        
        return u
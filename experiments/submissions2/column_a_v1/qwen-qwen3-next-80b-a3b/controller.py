import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        # Tuned gains for slow 194-min process with noise
        self.Kp = np.array([0.12, 0.12])   # Reduced P to avoid overshoot
        self.Ki = np.array([0.00015, 0.00015])  # Slower integral to prevent windup
        self.Kd = np.array([0.0, 0.0])     # No derivative - noise and delay
        self.max_actuator_rate = 0.0025    # Tighter slew limit to avoid duty breach
        self.L_min, self.L_max = 1.5, 4.5
        self.V_min, self.V_max = 2.0, 5.0
        self.reset()
    
    def reset(self):
        self.integral_error = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_output = np.array([2.70629, 3.20629])  # Start at nominal
        self.last_time = 0.0
        self.last_y = np.array([0.99, 0.99])
        self.last_D = 0.5  # Estimated initial draw
        self.last_B = 0.5  # Estimated initial draw
        self.safety_margin = 0.01  # Buffer for unmeasured draws
    
    def step(self, t, y, r, quality):
        # Handle bad measurements: use last known good
        if not quality[0]:
            y[0] = self.last_y[0]
        if not quality[1]:
            y[1] = self.last_y[1]
        
        # Setpoints: use last good if nan
        setpoint = np.where(np.isnan(r), self.last_y, r)
        
        # Calculate errors (direct acting)
        error = setpoint - y
        
        dt = t - self.last_time
        if dt <= 0:
            dt = self.sample_time
        
        # Anti-windup: only integrate if actuator is not saturated AND error is pushing away from saturation
        L_out, V_out = self.last_output
        L_sat = (L_out <= self.L_min + self.safety_margin and error[0] < 0) or \
                (L_out >= self.L_max - self.safety_margin and error[0] > 0)
        V_sat = (V_out <= self.V_min + self.safety_margin and error[1] < 0) or \
                (V_out >= self.V_max - self.safety_margin and error[1] > 0)
        
        if not L_sat:
            self.integral_error[0] += error[0] * dt
        if not V_sat:
            self.integral_error[1] += error[1] * dt
        
        # Proportional + Integral control
        P_term = self.Kp * error
        I_term = self.Ki * self.integral_error
        u_raw = P_term + I_term
        
        # Predict draws to enforce safety without measurement
        L_pred = self.last_output[0] + u_raw[0] * dt
        V_pred = self.last_output[1] + u_raw[1] * dt
        D_pred = V_pred - L_pred
        B_pred = L_pred + 1.0 - V_pred
        xB_pred = 1.0 - y[1]  # light key in bottoms
        
        # Safety check: if predicted violation, reduce control effort
        if D_pred < 0.05 + self.safety_margin or B_pred < 0.05 + self.safety_margin or \
           y[0] < 0.85 + self.safety_margin or xB_pred > 0.15 - self.safety_margin:
            # Reduce integral and proportional gain temporarily
            u_raw *= 0.3  # Aggressive attenuation for safety
        
        # Apply slew rate limiting
        u_new = self.last_output + u_raw
        u_new = np.clip(u_new, self.L_min, self.L_max)
        u_new[1] = np.clip(u_new[1], self.V_min, self.V_max)
        
        # Apply tighter slew rate limit
        max_delta = self.max_actuator_rate * dt
        u_new = np.clip(u_new, 
                        self.last_output - max_delta, 
                        self.last_output + max_delta)
        
        # Final safety: recompute draws from output and reject if unsafe
        L_final, V_final = u_new[0], u_new[1]
        D_final = V_final - L_final
        B_final = L_final + 1.0 - V_final
        xB_final = 1.0 - y[1]
        
        if D_final < 0.05 or B_final < 0.05 or y[0] < 0.85 or xB_final > 0.15:
            u_new = self.last_output  # Hold last safe value
        
        # Update state
        self.last_output = u_new
        self.last_error = error
        self.last_y = y
        self.last_time = t
        self.last_D = D_final
        self.last_B = B_final
        
        return u_new
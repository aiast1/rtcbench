import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Operating point
        self.L_nom = 2.70629
        self.V_nom = 3.20629
        self.F = 1.0
        self.zF = 0.5
        
        # PID parameters - further tuned for better tracking
        # Increased gains more aggressively while monitoring duty
        self.Kp_D = 2.2      # stronger proportional action
        self.Ki_D = 0.055    # faster integral response
        self.Kd_D = 0.0      # no derivative on noisy measurement
        
        self.Kp_B = 2.2      # stronger proportional action
        self.Ki_B = 0.055    # faster integral response
        self.Kd_B = 0.0
        
        # Integral state
        self.int_err_D = 0.0
        self.int_err_B = 0.0
        
        # Anti-windup limits - increased to allow more aggressive control
        self.int_limit = 1.2
        
        # Actuator limits
        self.L_min, self.L_max = 1.5, 4.5
        self.V_min, self.V_max = 2.0, 5.0
        
        # Product draw safety
        self.D_min = 0.05
        self.B_min = 0.05
        self.purity_min = 0.85
        self.xB_max = 0.15
        
        # Tracking for duty limit (0.003 kmol/min per second)
        self.duty_limit = 0.003
        self.last_L = self.L_nom
        self.last_V = self.V_nom
        
        # Low-pass filter for noisy measurements - faster response
        self.alpha_filter = 0.5
        self.filt_D = 0.99
        self.filt_B = 0.99
        
    def reset(self):
        self.int_err_D = 0.0
        self.int_err_B = 0.0
        self.last_L = self.L_nom
        self.last_V = self.V_nom
        self.filt_D = 0.99
        self.filt_B = 0.99
    
    def step(self, t, y, r, quality):
        # Extract measurements
        yD_raw = y[0]  # AT-101: distillate purity
        yB_raw = y[1]  # AT-102: bottoms purity (heavy key)
        
        # Apply low-pass filter to reduce noise impact
        if quality[0]:
            self.filt_D = self.alpha_filter * yD_raw + (1 - self.alpha_filter) * self.filt_D
        yD = self.filt_D
        
        if quality[1]:
            self.filt_B = self.alpha_filter * yB_raw + (1 - self.alpha_filter) * self.filt_B
        yB = self.filt_B
        
        # Setpoints
        r_D = r[0] if not np.isnan(r[0]) else 0.99
        r_B = r[1] if not np.isnan(r[1]) else 0.99
        
        # Errors
        err_D = r_D - yD
        err_B = r_B - yB
        
        # Integral action with anti-windup
        self.int_err_D += err_D * self.sample_time
        self.int_err_B += err_B * self.sample_time
        
        # Clamp integral to prevent windup
        self.int_err_D = np.clip(self.int_err_D, -self.int_limit, self.int_limit)
        self.int_err_B = np.clip(self.int_err_B, -self.int_limit, self.int_limit)
        
        # PID outputs (no derivative on noisy measurement)
        pid_D = self.Kp_D * err_D + self.Ki_D * self.int_err_D
        pid_B = self.Kp_B * err_B + self.Ki_B * self.int_err_B
        
        # Decoupling: distillate purity primarily controlled by reflux L
        # Bottoms purity primarily controlled by boilup V
        L_cmd = self.L_nom + pid_D
        V_cmd = self.V_nom + pid_B
        
        # Enforce actuator limits first
        L_cmd = np.clip(L_cmd, self.L_min, self.L_max)
        V_cmd = np.clip(V_cmd, self.V_min, self.V_max)
        
        # Enforce duty limit (slew rate constraint) - smooth changes
        max_change = self.duty_limit * self.sample_time
        L_cmd = np.clip(L_cmd, self.last_L - max_change, self.last_L + max_change)
        V_cmd = np.clip(V_cmd, self.last_V - max_change, self.last_V + max_change)
        
        # Ensure minimum product draws to avoid trip
        # D = V - L, B = L + F - V
        D_implied = V_cmd - L_cmd
        B_implied = L_cmd + self.F - V_cmd
        
        # Adjust to maintain minimum draws
        if D_implied < self.D_min:
            V_cmd = min(self.V_max, L_cmd + self.D_min)
        if B_implied < self.B_min:
            V_cmd = max(self.V_min, L_cmd + self.F - self.B_min)
        
        # Re-enforce limits after draw constraints
        L_cmd = np.clip(L_cmd, self.L_min, self.L_max)
        V_cmd = np.clip(V_cmd, self.V_min, self.V_max)
        
        # Final safety check: ensure draws stay above minimum
        D_final = V_cmd - L_cmd
        B_final = L_cmd + self.F - V_cmd
        
        if D_final < self.D_min or B_final < self.B_min:
            # Fall back to nominal if constraints can't be satisfied
            L_cmd = self.L_nom
            V_cmd = self.V_nom
        
        # Store for next cycle
        self.last_L = L_cmd
        self.last_V = V_cmd
        
        return np.array([L_cmd, V_cmd])
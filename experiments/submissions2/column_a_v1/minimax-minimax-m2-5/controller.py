import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        # Initial actuator values (steady-state operating point)
        self.L_init = 2.70629
        self.V_init = 3.20629
        # Tuned for better tracking while maintaining safety
        self.Kp_yD = 0.05
        self.Kp_xB = 0.05
        self.Ki_yD = 0.0003
        self.Ki_xB = 0.0003
        # Actuator duty cycle limit
        self.max_change = 0.003
        # Hard limits for actuators
        self.L_min = 1.5
        self.L_max = 4.5
        self.V_min = 2.0
        self.V_max = 5.0

    def reset(self):
        self.L = self.L_init
        self.V = self.V_init
        self.prev_L = self.L
        self.prev_V = self.V
        self.integral_L = 0.0
        self.integral_V = 0.0
        self.last_yD = 0.99
        self.last_xB = 0.99
        self.last_D = 0.5
        self.last_B = 0.5

    def step(self, t, y, r, quality):
        # Update measurements only when fresh
        if quality[0]:
            self.last_yD = y[0]
        if quality[1]:
            self.last_xB = y[1]

        # Estimate product flows from previous actuator values
        D = self.prev_V - self.prev_L
        B = self.prev_L + 1.0 - self.prev_V
        self.last_D = D
        self.last_B = B

        # Compute tracking errors
        e_yD = r[0] - self.last_yD
        e_xB = r[1] - self.last_xB

        # Proportional actions
        p_L = self.Kp_yD * e_yD
        p_V = self.Kp_xB * e_xB

        # Integral actions with anti-windup
        dt = self.sample_time
        if quality[0]:
            if self.prev_L > self.L_min + 1e-6 and self.prev_L < self.L_max - 1e-6:
                self.integral_L += self.Ki_yD * e_yD * dt
        if quality[1]:
            if self.prev_V > self.V_min + 1e-6 and self.prev_V < self.V_max - 1e-6:
                self.integral_V += self.Ki_xB * e_xB * dt

        # Compute desired changes
        delta_L = p_L + self.integral_L
        delta_V = p_V + self.integral_V

        # Compute desired actuator values
        L_desired = self.prev_L + delta_L
        V_desired = self.prev_V + delta_V

        # Conservative product flow constraints: keep margins away from limits
        # D = V - L >= 0.10 (use 0.10 instead of 0.05 for safety margin)
        # B = L + 1 - V >= 0.10
        min_D = 0.10
        min_B = 0.10
        
        if V_desired < L_desired + min_D:
            V_desired = L_desired + min_D
        if V_desired > L_desired + 1.0 - min_B:
            V_desired = L_desired + 1.0 - min_B

        # Enforce hard actuator limits
        L_desired = np.clip(L_desired, self.L_min, self.L_max)
        V_desired = np.clip(V_desired, self.V_min, self.V_max)

        # Apply rate limiting
        L_change = np.clip(L_desired - self.prev_L, -self.max_change, self.max_change)
        V_change = np.clip(V_desired - self.prev_V, -self.max_change, self.max_change)

        L = self.prev_L + L_change
        V = self.prev_V + V_change

        # Final safety check: ensure product flows are safe
        D_final = V - L
        B_final = L + 1.0 - V
        
        # Hard clamp to ensure safety
        if D_final < 0.05:
            V = L + 0.05
            D_final = 0.05
        if B_final < 0.05:
            V = L + 1.0 - 0.05
            B_final = 0.05

        self.prev_L = L
        self.prev_V = V

        return np.array([L, V])
import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.y3_min = -0.5  # Safety constraint: y3 >= -0.5
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        self.int_state = np.zeros(3)
        self.prev_u = np.zeros(3)
        self.tau_aw = 10.0  # Anti-windup time constant
        
        # Tuned PI gains (conservative for robustness across uncertainty)
        self.Kc = np.array([0.08, 0.08, 0.12])  # Proportional gains
        self.tauI = np.array([40.0, 40.0, 30.0])  # Integral time constants

    def reset(self):
        self.int_state = np.zeros(3)
        self.prev_u = np.zeros(3)

    def step(self, t, y, r, quality):
        # Compute error for scored channels (ignore NaN in r)
        error = np.zeros(3)
        for i in range(3):
            if not np.isnan(r[i]):
                error[i] = r[i] - y[i]
        
        # Safety override for bottoms reflux temperature (y[2])
        safety_error = max(0.0, self.y3_min - y[2])  # Positive if below limit
        K_safety = 2.5  # Aggressive but stable safety gain
        safety_u2 = K_safety * safety_error
        
        # Compute tentative feedback control (PI only)
        u_fb = np.zeros(3)
        for i in range(3):
            if self.tauI[i] > 0:
                # PI control: u = Kc * error + (Kc/tauI) * integral_error
                p_term = self.Kc[i] * error[i]
                i_term = (self.Kc[i] / self.tauI[i]) * self.int_state[i]
                u_fb[i] = p_term + i_term
            else:
                u_fb[i] = self.Kc[i] * error[i]  # P-only if no integral
        
        # Add safety override to u2
        u_raw = u_fb.copy()
        u_raw[2] += safety_u2
        
        # Apply actuator limits
        u = np.clip(u_raw, self.u_min, self.u_max)
        
        # Anti-windup: back-calculation method
        for i in range(3):
            if self.tauI[i] > 0 and self.Kc[i] != 0:
                # Update integral state: 
                #   int_state_dot = error + (tauI/(tau_aw*Kc)) * (u_actual - u_unsaturated)
                int_state_update = self.sample_time * (
                    error[i] + 
                    (self.tauI[i] / (self.tau_aw * self.Kc[i])) * 
                    (u[i] - u_raw[i])
                )
                self.int_state[i] += int_state_update
        
        self.prev_u = u.copy()
        return u
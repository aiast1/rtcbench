import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp_T = 100.0
        self.Ki_T = 0.1
        self.Kp_C = 0.5
        self.Ki_C = 0.001
        self.dt = brief.sample_time
        self.T_safety_threshold = 145.0
        self.alpha_T = 0.3  # EMA smoothing factor for temperature

    def reset(self):
        self.I_term_C = 14.19
        self.last_C = 1.09
        self.I_term_T = -1113.5
        self.last_T_filtered = 114.19
        self.last_T_raw = 114.19

    def step(self, t, y, r, quality):
        # Process temperature measurement with EMA filter
        if quality[1]:
            self.last_T_raw = y[1]
            self.last_T_filtered = (self.alpha_T * y[1] 
                                   + (1 - self.alpha_T) * self.last_T_filtered)
        current_T = self.last_T_filtered
        
        # Concentration control with quality-aware integration
        if quality[0]:
            self.last_C = y[0]
            e_C = r[0] - self.last_C
            self.I_term_C += self.Ki_C * e_C * self.dt
        else:
            e_C = r[0] - self.last_C  # Use last good measurement for error
            
        P_C = self.Kp_C * e_C
        u_C = np.clip(P_C + self.I_term_C, 3.0, 35.0)
        # Anti-windup: only allow I-term to accumulate when not saturated
        if (P_C + self.I_term_C) != u_C:
            self.I_term_C = u_C - P_C

        # Temperature control with safety override
        e_T = r[1] - current_T
        if quality[1]:
            self.I_term_T += self.Ki_T * e_T * self.dt
            
        if current_T > self.T_safety_threshold:
            u_T = -9000.0
            self.I_term_T = u_T - self.Kp_T * e_T  # Anti-windup reset
        else:
            P_T = self.Kp_T * e_T
            u_T_unclamped = P_T + self.I_term_T
            u_T = np.clip(u_T_unclamped, -9000.0, 0.0)
            # Anti-windup adjustment
            if u_T != u_T_unclamped:
                self.I_term_T = u_T - P_T

        return np.array([u_C, u_T])
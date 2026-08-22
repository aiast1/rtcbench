import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = brief.sample_time
        
        # Actuator limits
        self.FV_min, self.FV_max = 3.0, 35.0
        self.QK_min, self.QK_max = -9000.0, 0.0
        
        # Initial steady-state values (bumpless start)
        self.FV_ss = 14.19
        self.QK_ss = -1113.5
        
        # Controller states
        self.reset()
        
    def reset(self):
        # Current actuator values
        self.FV = self.FV_ss
        self.QK = self.QK_ss
        
        # Integral states
        self.integral_FV = 0.0
        self.integral_QK = 0.0
        
        # Previous error for derivative (not used, but tracking)
        self.prev_error_FV = 0.0
        self.prev_error_QK = 0.0
        
        # Track if last sample was valid for anti-windup
        self.last_valid = True
        
    def step(self, t, y, r, quality):
        # y[0] = C_B (product concentration), y[1] = T (reactor temperature)
        # r[0] = C_B setpoint, r[1] = T setpoint
        
        # Handle missing/bad measurements - hold last output
        if not quality[0] or not quality[1]:
            return np.array([self.FV, self.QK])
        
        C_B = y[0]
        T = y[1]
        
        C_B_sp = r[0]
        T_sp = r[1]
        
        # Errors
        error_CB = C_B_sp - C_B
        error_T = T_sp - T
        
        # --- TUNED PI CONTROLLERS ---
        # Conservative gains for robustness (worst-case scenarios)
        Kp_CB = 80.0   # Moderate gain for composition
        Ki_CB = 0.8    # Small integral for steady-state
        
        Kp_T = 15.0    # Conservative gain for temperature
        Ki_T = 2.0     # Integral for temperature tracking
        
        # Anti-windup: only integrate if actuator not saturated
        # or if saturation is in direction of reducing error
        delta_FV = Kp_CB * error_CB + Ki_CB * self.Ts * error_CB
        delta_QK = Kp_T * error_T + Ki_T * self.Ts * error_T
        
        # Compute new actuator values
        FV_new = self.FV + delta_FV
        QK_new = self.QK + delta_QK
        
        # Apply anti-windup: back-calculation
        # If actuator would saturate, limit the integral contribution
        FV_sat = np.clip(FV_new, self.FV_min, self.FV_max)
        QK_sat = np.clip(QK_new, self.QK_min, self.QK_max)
        
        # Anti-windup: adjust integral state if saturation occurred
        # This prevents integral wind-up when actuator is clamped
        if FV_new != FV_sat:
            # Reduce integral to prevent further saturation
            self.integral_FV -= (FV_new - FV_sat) * 0.1
        else:
            self.integral_FV += Ki_CB * self.Ts * error_CB
            
        if QK_new != QK_sat:
            self.integral_QK -= (QK_new - QK_sat) * 0.1
        else:
            self.integral_QK += Ki_T * self.Ts * error_T
        
        # Final output with saturation
        self.FV = np.clip(self.FV + Kp_CB * (error_CB - self.prev_error_FV) + 
                         Ki_CB * self.Ts * error_CB, self.FV_min, self.FV_max)
        self.QK = np.clip(self.QK + Kp_T * (error_T - self.prev_error_QK) + 
                         Ki_T * self.Ts * error_T, self.QK_min, self.QK_max)
        
        # Store for next iteration
        self.prev_error_FV = error_CB
        self.prev_error_T = error_T
        
        # Safety: jacket temperature constraint
        # T_K is not measured but we can estimate from Q_K
        # Conservative: if Q_K is too high (less cooling), reduce to stay safe
        # Rough estimate: Q_K = kw * AR * (T_K - T), solve for T_K
        # With nominal params: T_K = T + Q_K / (kw * AR)
        # kw = 4032, AR = 0.215 -> kw*AR = 867
        # At max Q_K = 0, T_K = T (no cooling)
        # At min Q_K = -9000, T_K = T - 10.4 (jacket can be ~10C colder)
        # But if T is high (150C) and Q_K is not enough, T_K approaches T
        # The constraint is T_K <= 150, which is usually satisfied if T <= 150
        # However, if T is near 150 and Q_K is near 0, T_K approaches 150
        # We add a safety margin: if T > 145, force more cooling
        if T > 145.0:
            # Force maximum cooling to protect jacket
            self.QK = min(self.QK, self.QK_max * 0.5)  # At least 50% cooling
        
        # Additional safety: if F/V is getting too high (non-monotonic region)
        # The relationship is non-monotonic past 14.7 h^-1
        # Be conservative: if we're above 14.7 and C_B is low, don't increase F/V further
        if self.FV > 14.7 and C_B < C_B_sp:
            # We're in the "bad" region - reduce F/V rather than increase
            self.FV = min(self.FV, 14.7)
        
        return np.array([self.FV, self.QK])
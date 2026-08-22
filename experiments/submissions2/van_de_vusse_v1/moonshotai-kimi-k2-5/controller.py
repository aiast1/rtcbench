import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time
        
        # Actuator limits
        self.u_min = np.array([3.0, -9000.0])
        self.u_max = np.array([35.0, 0.0])
        
        # Initial steady-state values
        self.u_ss = np.array([14.19, -1113.5])
        
        # Measurement indices
        self.idx_CB = 0
        self.idx_T = 1
        
        # Actuator indices
        self.idx_FV = 0
        self.idx_QK = 1
        
        # Setpoint indices
        self.idx_sp_CB = 0
        self.idx_sp_T = 1
        
        # PID gains - tuned for better balance between tracking and robustness
        # CB loop: more conservative to avoid oscillations on difficult seeds
        self.Kp_CB = 3.0
        self.Ki_CB = 0.08
        self.Kd_CB = 5.0
        
        # T loop: moderate gains
        self.Kp_T = -120.0
        self.Ki_T = -3.0
        self.Kd_T = -300.0
        
        # Derivative filter
        self.tau_f = 4.0
        
        # Anti-windup
        self.Kaw = 0.8
        
        # Safety limits
        self.T_max = 150.0
        
        # Operating point
        self.CB_op = 1.09
        self.T_op = 114.19
        
        self.reset()
    
    def reset(self):
        self.u = self.u_ss.copy()
        self.int_CB = 0.0
        self.int_T = 0.0
        
        self.y_CB_prev = self.CB_op
        self.y_T_prev = self.T_op
        self.dy_f_CB = 0.0
        self.dy_f_T = 0.0
        
        self.r_CB_prev = self.CB_op
        self.r_T_prev = self.T_op
        
        self.u_prev = self.u_ss.copy()
        self.first_step = True
        
        # Exponential moving average for filtering
        self.CB_filt = self.CB_op
        self.T_filt = self.T_op
        self.alpha_filt = 0.3
        
        # Setpoint rate limiting
        self.r_CB_filt = self.CB_op
        self.r_T_filt = self.T_op
    
    def step(self, t, y, r, quality):
        # Extract measurements with quality handling
        CB_raw = y[self.idx_CB] if quality[self.idx_CB] else self.CB_filt
        T_raw = y[self.idx_T] if quality[self.idx_T] else self.T_filt
        
        # Exponential filtering
        self.CB_filt = (1 - self.alpha_filt) * self.CB_filt + self.alpha_filt * CB_raw
        self.T_filt = (1 - self.alpha_filt) * self.T_filt + self.alpha_filt * T_raw
        
        CB_meas = self.CB_filt
        T_meas = self.T_filt
        
        # Extract setpoints with gentle filtering for smooth transitions
        r_CB_target = r[self.idx_sp_CB] if not np.isnan(r[self.idx_sp_CB]) else CB_meas
        r_T_target = r[self.idx_sp_T] if not np.isnan(r[self.idx_sp_T]) else T_meas
        
        # Smooth setpoint changes
        sp_alpha = 0.15
        self.r_CB_filt = (1 - sp_alpha) * self.r_CB_filt + sp_alpha * r_CB_target
        self.r_T_filt = (1 - sp_alpha) * self.r_T_filt + sp_alpha * r_T_target
        
        r_CB = self.r_CB_filt
        r_T = self.r_T_filt
        
        # Safety check with early action
        T_safety_margin = 10.0
        T_emergency = T_meas > (self.T_max - T_safety_margin)
        T_warning = T_meas > (self.T_max - 15.0)
        
        # Compute filtered derivatives
        alpha = self.dt / (self.tau_f + self.dt)
        
        if self.first_step:
            self.y_CB_prev = CB_meas
            self.y_T_prev = T_meas
            self.dy_f_CB = 0.0
            self.dy_f_T = 0.0
            self.first_step = False
        else:
            dy_CB = (CB_meas - self.y_CB_prev) / self.dt
            dy_T = (T_meas - self.y_T_prev) / self.dt
            
            self.dy_f_CB = (1 - alpha) * self.dy_f_CB + alpha * dy_CB
            self.dy_f_T = (1 - alpha) * self.dy_f_T + alpha * dy_T
        
        self.y_CB_prev = CB_meas
        self.y_T_prev = T_meas
        
        # Compute errors
        e_CB = r_CB - CB_meas
        e_T = r_T - T_meas
        
        # PID for CB (F/V)
        P_CB = self.Kp_CB * e_CB
        I_CB = self.int_CB
        D_CB = -self.Kd_CB * self.dy_f_CB
        
        u_CB_raw = self.u_ss[self.idx_FV] + P_CB + I_CB + D_CB
        
        # PID for T (Q_K)
        P_T = self.Kp_T * e_T
        I_T = self.int_T
        D_T = -self.Kd_T * self.dy_f_T
        
        u_T_raw = self.u_ss[self.idx_QK] + P_T + I_T + D_T
        
        # Safety overrides
        if T_emergency:
            u_T_raw = -9000.0
            u_CB_raw = max(u_CB_raw * 0.8, 6.0)
        elif T_warning:
            # Gradual increase in cooling
            u_T_raw = min(u_T_raw, -2000.0)
        
        # Clamp
        u_CB_sat = np.clip(u_CB_raw, self.u_min[self.idx_FV], self.u_max[self.idx_FV])
        u_T_sat = np.clip(u_T_raw, self.u_min[self.idx_QK], self.u_max[self.idx_QK])
        
        # Anti-windup with conditional integration
        if u_CB_raw != u_CB_sat:
            # Saturated - only back-calculate
            self.int_CB += self.Kaw * (u_CB_sat - u_CB_raw) * 0.5
        else:
            self.int_CB += self.Ki_CB * e_CB * self.dt
        
        if u_T_raw != u_T_sat:
            self.int_T += self.Kaw * (u_T_sat - u_T_raw) * 0.5
        else:
            self.int_T += self.Ki_T * e_T * self.dt
        
        # Clamp integrals
        self.int_CB = np.clip(self.int_CB, -12.0, 12.0)
        self.int_T = np.clip(self.int_T, -6000.0, 6000.0)
        
        # Moderate rate limiting
        max_rate = np.array([2.5, 600.0])
        du = np.array([u_CB_sat, u_T_sat]) - self.u_prev
        du_clamped = np.clip(du, -max_rate, max_rate)
        u_out = self.u_prev + du_clamped
        
        u_out = np.clip(u_out, self.u_min, self.u_max)
        self.u_prev = u_out.copy()
        
        return u_out
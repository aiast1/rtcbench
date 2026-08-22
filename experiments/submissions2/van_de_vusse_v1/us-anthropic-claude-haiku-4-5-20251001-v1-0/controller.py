import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # State for integral action and anti-windup
        self.integral_cb = 0.0
        self.integral_t = 0.0
        
        # Previous measurement for derivative (with filtering)
        self.prev_cb = None
        self.prev_t = None
        self.filtered_dcb_dt = 0.0
        self.filtered_dt_dt = 0.0
        
        # PID gains - tuned more aggressively for better tracking
        # CB channel: product concentration
        self.kp_cb = 1.2
        self.ki_cb = 0.25
        self.kd_cb = 0.08
        
        # T channel: reactor temperature
        self.kp_t = 120.0
        self.ki_t = 25.0
        self.kd_t = 8.0
        
        # Derivative filter time constant (low-pass)
        self.tau_d = 1.5
        self.alpha_d = self.dt / (self.tau_d + self.dt)
        
        # Actuator limits
        self.fv_min = 3.0
        self.fv_max = 35.0
        self.qk_min = -9000.0
        self.qk_max = 0.0
        
        # Temperature limits (safety)
        self.t_max = 150.0
        self.tk_max = 150.0
        
        # Nominal operating point for initialization
        self.fv_nom = 14.19
        self.qk_nom = -1113.5
        
        # Anti-windup limits - more generous to allow faster response
        self.integral_cb_max = 8.0
        self.integral_t_max = 5000.0
        
        # Track last good measurement for stale data handling
        self.last_good_cb = 0.5
        self.last_good_t = 114.19
        
        # Actuator travel tracking for duty limit
        self.last_fv = self.fv_nom
        self.last_qk = self.qk_nom
        self.total_travel = 0.0
        
    def reset(self):
        self.integral_cb = 0.0
        self.integral_t = 0.0
        self.prev_cb = None
        self.prev_t = None
        self.filtered_dcb_dt = 0.0
        self.filtered_dt_dt = 0.0
        self.last_good_cb = 0.5
        self.last_good_t = 114.19
        self.last_fv = self.fv_nom
        self.last_qk = self.qk_nom
        self.total_travel = 0.0
        
    def step(self, t, y, r, quality):
        # Extract measurements
        cb_meas = y[0]
        t_meas = y[1]
        
        # Handle stale/bad measurements
        if quality[0]:
            cb = cb_meas
            self.last_good_cb = cb
        else:
            cb = self.last_good_cb
            
        if quality[1]:
            t_reactor = t_meas
            self.last_good_t = t_reactor
        else:
            t_reactor = self.last_good_t
        
        # Extract setpoints
        cb_sp = r[0]
        t_sp = r[1]
        
        # Compute errors
        err_cb = cb_sp - cb
        err_t = t_sp - t_reactor
        
        # Derivative action with low-pass filtering (only on good measurements)
        if self.prev_cb is not None and quality[0]:
            dcb_dt_raw = (cb - self.prev_cb) / self.dt
            self.filtered_dcb_dt = (1.0 - self.alpha_d) * self.filtered_dcb_dt + self.alpha_d * dcb_dt_raw
        
        if self.prev_t is not None and quality[1]:
            dt_dt_raw = (t_reactor - self.prev_t) / self.dt
            self.filtered_dt_dt = (1.0 - self.alpha_d) * self.filtered_dt_dt + self.alpha_d * dt_dt_raw
        
        self.prev_cb = cb
        self.prev_t = t_reactor
        
        # Integral action with anti-windup
        self.integral_cb += err_cb * self.dt
        self.integral_cb = np.clip(self.integral_cb, -self.integral_cb_max, self.integral_cb_max)
        
        self.integral_t += err_t * self.dt
        self.integral_t = np.clip(self.integral_t, -self.integral_t_max, self.integral_t_max)
        
        # PID outputs (before saturation)
        pid_cb = (self.kp_cb * err_cb + 
                  self.ki_cb * self.integral_cb - 
                  self.kd_cb * self.filtered_dcb_dt)
        
        pid_t = (self.kp_t * err_t + 
                 self.ki_t * self.integral_t - 
                 self.kd_t * self.filtered_dt_dt)
        
        # Map PID outputs to actuators
        # CB error -> FV adjustment (positive error means increase feed)
        # T error -> QK adjustment (positive error means cool more, i.e., more negative QK)
        
        fv_cmd = self.fv_nom + pid_cb
        qk_cmd = self.qk_nom + pid_t
        
        # Apply hard limits
        fv_cmd = np.clip(fv_cmd, self.fv_min, self.fv_max)
        qk_cmd = np.clip(qk_cmd, self.qk_min, self.qk_max)
        
        # Track actuator travel for duty limit awareness
        fv_travel = abs(fv_cmd - self.last_fv)
        qk_travel = abs(qk_cmd - self.last_qk)
        self.total_travel += fv_travel + qk_travel
        
        self.last_fv = fv_cmd
        self.last_qk = qk_cmd
        
        # Anti-windup: if saturated, reduce integral
        if fv_cmd == self.fv_min or fv_cmd == self.fv_max:
            self.integral_cb *= 0.90
        
        if qk_cmd == self.qk_min or qk_cmd == self.qk_max:
            self.integral_t *= 0.90
        
        return np.array([fv_cmd, qk_cmd])
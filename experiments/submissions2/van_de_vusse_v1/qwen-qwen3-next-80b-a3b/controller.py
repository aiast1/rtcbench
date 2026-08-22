import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.dt = brief.sample_time
        self.n_actuators = 2
        self.n_measurements = 2
        
        # PID parameters (tuned for robustness, not peak performance)
        # C_B controller (setpoint: [0] - product concentration)
        self.Kp_CB = 0.15
        self.Ki_CB = 0.001
        self.Kd_CB = 0.01
        
        # T controller (setpoint: [1] - reactor temperature)
        self.Kp_T = 0.8
        self.Ki_T = 0.02
        self.Kd_T = 0.05
        
        # Anti-windup and constraints
        self.aw_factor = 0.5  # anti-windup gain
        self.max_actuator_change = 0.5  # limit per step to avoid chattering
        self.slew_limit_FV = 0.5  # max change in F/V per step (1/h)
        self.slew_limit_QK = 500.0  # max change in Q_K per step (kJ/h)
        
        # Actuator limits
        self.FV_min, self.FV_max = 3.0, 35.0
        self.QK_min, self.QK_max = -9000.0, 0.0
        
        # State variables
        self.integral_CB = 0.0
        self.integral_T = 0.0
        self.prev_error_CB = 0.0
        self.prev_error_T = 0.0
        self.prev_output_FV = 14.19
        self.prev_output_QK = -1113.5
        self.prev_T = 114.19
        self.prev_CB = 1.09
        
        # Filter for derivative (alpha for low-pass filter)
        self.alpha_d = 0.1
        
        # For temperature constraint (T_K estimation)
        self.TK_estimate = 114.19  # initial estimate based on steady state
        
        # Setpoint schedule (t, [CB_setpoint, T_setpoint])
        self.setpoints = [
            (0.0, np.array([1.09, 114.19])),
            (900.0, np.array([0.928, 114.19])),
            (2900.0, np.array([1.081, 114.19])),
            (4600.0, np.array([0.892, 114.19])),
            (6100.0, np.array([1.09, 114.19]))
        ]
        
        # Initialize current setpoint
        self.current_setpoint = np.array([1.09, 114.19])
        self.next_setpoint_idx = 1
        
    def reset(self):
        self.integral_CB = 0.0
        self.integral_T = 0.0
        self.prev_error_CB = 0.0
        self.prev_error_T = 0.0
        self.prev_output_FV = 14.19
        self.prev_output_QK = -1113.5
        self.prev_T = 114.19
        self.prev_CB = 1.09
        self.TK_estimate = 114.19
        self.current_setpoint = np.array([1.09, 114.19])
        self.next_setpoint_idx = 1
        
    def _update_setpoint(self, t):
        """Update setpoint based on time schedule"""
        if self.next_setpoint_idx < len(self.setpoints):
            t_next, sp_next = self.setpoints[self.next_setpoint_idx]
            if t >= t_next:
                # Check if it's a ramp
                if self.next_setpoint_idx == 2:  # ramp from 2900s to 3500s
                    ramp_start_t, ramp_start_sp = self.setpoints[1]
                    ramp_end_t, ramp_end_sp = self.setpoints[2]
                    if t < ramp_end_t:
                        alpha = (t - ramp_start_t) / (ramp_end_t - ramp_start_t)
                        self.current_setpoint = ramp_start_sp + alpha * (ramp_end_sp - ramp_start_sp)
                    else:
                        self.current_setpoint = ramp_end_sp
                        self.next_setpoint_idx += 1
                else:
                    self.current_setpoint = sp_next
                    self.next_setpoint_idx += 1
    
    def _estimate_jacket_temp(self, FV, QK, T, CB):
        """Simple thermal model to estimate jacket temperature T_K"""
        # Based on energy balance: Q_K = kw * AR * (T_K - T)
        # So T_K = T + Q_K / (kw * AR)
        # Using nominal values from model hint
        kw = 4032.0
        AR = 0.215
        # Estimate T_K from current heat duty and reactor temp
        T_K_est = T + QK / (kw * AR)
        # Clamp to safety limit
        T_K_est = min(T_K_est, 150.0)
        return T_K_est
    
    def step(self, t, y, r, quality):
        # Update setpoint schedule
        self._update_setpoint(t)
        
        # Use provided setpoints if valid, otherwise use schedule
        if not np.isnan(r[0]):
            CB_sp = r[0]
        else:
            CB_sp = self.current_setpoint[0]
            
        if not np.isnan(r[1]):
            T_sp = r[1]
        else:
            T_sp = self.current_setpoint[1]
        
        # Extract measurements
        CB_meas = y[0]
        T_meas = y[1]
        
        # Check quality flags - if bad, use previous value
        if not quality[0]:
            CB_meas = self.prev_CB
        if not quality[1]:
            T_meas = self.prev_T
            
        # Update previous values
        self.prev_CB = CB_meas
        self.prev_T = T_meas
        
        # Calculate errors
        error_CB = CB_sp - CB_meas
        error_T = T_sp - T_meas
        
        # Update integral terms with anti-windup
        # Only integrate if actuators are not saturated or if error points away from saturation
        # For FV (actuator 0): we're trying to increase CB -> higher FV might help or hurt (non-monotonic)
        # For QK (actuator 1): cooling - more negative = more cooling
        
        # Anti-windup for CB controller (FV actuator)
        # If FV is at limit and error is in direction that would require more movement, disable integral
        FV_desired = self.prev_output_FV + self.Kp_CB * error_CB + self.Ki_CB * error_CB * self.dt
        if FV_desired > self.FV_max and error_CB > 0:
            # Integral windup: error is positive but FV is maxed out
            integral_term_CB = self.integral_CB
        elif FV_desired < self.FV_min and error_CB < 0:
            # Integral windup: error is negative but FV is mined out
            integral_term_CB = self.integral_CB
        else:
            integral_term_CB = self.integral_CB + error_CB * self.dt
        
        # Anti-windup for T controller (QK actuator)
        QK_desired = self.prev_output_QK + self.Kp_T * error_T + self.Ki_T * error_T * self.dt
        if QK_desired < self.QK_min and error_T > 0:  # need more cooling but at min
            integral_term_T = self.integral_T
        elif QK_desired > self.QK_max and error_T < 0:  # need less cooling but at max (0)
            integral_term_T = self.integral_T
        else:
            integral_term_T = self.integral_T + error_T * self.dt
        
        # Derivative terms with filtering
        d_CB = self.alpha_d * (error_CB - self.prev_error_CB) / self.dt + (1 - self.alpha_d) * self.prev_error_CB
        d_T = self.alpha_d * (error_T - self.prev_error_T) / self.dt + (1 - self.alpha_d) * self.prev_error_T
        
        # PID outputs
        u_CB = self.Kp_CB * error_CB + self.Ki_CB * integral_term_CB + self.Kd_CB * d_CB
        u_T = self.Kp_T * error_T + self.Ki_T * integral_term_T + self.Kd_T * d_T
        
        # Update integrals for next step
        self.integral_CB = integral_term_CB
        self.integral_T = integral_term_T
        self.prev_error_CB = error_CB
        self.prev_error_T = error_T
        
        # Combine outputs to actuators
        # FV (feed dilution) is primarily controlled by CB
        # QK (jacket heat) is primarily controlled by T
        # But they are coupled - so we use a simple decoupling
        FV_out = self.prev_output_FV + u_CB
        QK_out = self.prev_output_QK + u_T
        
        # Apply slew rate limits
        FV_out = np.clip(FV_out, 
                         self.prev_output_FV - self.slew_limit_FV, 
                         self.prev_output_FV + self.slew_limit_FV)
        QK_out = np.clip(QK_out, 
                         self.prev_output_QK - self.slew_limit_QK, 
                         self.prev_output_QK + self.slew_limit_QK)
        
        # Apply hard actuator limits
        FV_out = np.clip(FV_out, self.FV_min, self.FV_max)
        QK_out = np.clip(QK_out, self.QK_min, self.QK_max)
        
        # Estimate jacket temperature for safety
        self.TK_estimate = self._estimate_jacket_temp(FV_out, QK_out, T_meas, CB_meas)
        
        # Safety check - if T_K exceeds limit, reduce cooling demand
        if self.TK_estimate > 150.0:
            # Reduce QK (make it more negative) to cool more
            # But we're already at max cooling? Then we need to reduce FV to reduce reaction rate
            # Priority: reduce FV first (to reduce heat generation), then reduce QK if needed
            FV_out = max(self.FV_min, FV_out - 0.5)
            QK_out = max(self.QK_min, QK_out - 200.0)
            # Re-estimate
            self.TK_estimate = self._estimate_jacket_temp(FV_out, QK_out, T_meas, CB_meas)
            if self.TK_estimate > 150.0:
                # If still over, reduce FV further
                FV_out = max(self.FV_min, FV_out - 0.5)
                QK_out = max(self.QK_min, QK_out - 200.0)
        
        # Final output
        output = np.array([FV_out, QK_out])
        
        # Update previous outputs
        self.prev_output_FV = FV_out
        self.prev_output_QK = QK_out
        
        return output
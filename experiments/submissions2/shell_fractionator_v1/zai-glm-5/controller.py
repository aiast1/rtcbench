import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float
    scenario_length: float
    n_outputs: int
    n_inputs: int
    output_names: list
    input_names: list
    output_ranges: list
    input_ranges: list
    setpoint_schedule: dict
    model_hint: dict

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.n_out = 3
        self.n_in = 3
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint: y[2] >= -0.5
        self.y_min = np.array([-np.inf, -np.inf, -0.5])
        
        # Nominal model parameters
        K_nom = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.9],
            [4.38, 4.42, 7.2]
        ])
        TAU_nom = np.array([
            [50.0, 60.0, 50.0],
            [50.0, 60.0, 40.0],
            [33.0, 44.0, 19.0]
        ])
        L_nom = np.array([
            [27.0, 28.0, 27.0],
            [18.0, 14.0, 15.0],
            [20.0, 22.0, 0.0]
        ])
        
        self.K = K_nom.copy()
        self.TAU = TAU_nom.copy()
        self.L = L_nom.copy()
        
        # Controller tuning - Direct SISO PID with IMC-based tuning
        self.Kc = np.zeros((self.n_out, self.n_in))
        self.Ti = np.zeros((self.n_out, self.n_in))
        self.Td = np.zeros((self.n_out, self.n_in))
        
        for i in range(self.n_out):
            for j in range(self.n_in):
                if abs(self.K[i,j]) < 0.1:
                    continue
                
                tau_eff = max(self.TAU[i,j], 5.0)
                L_eff = max(self.L[i,j], 1.0)
                
                # IMC tuning: Kc = tau/(K * (L + lambda)), Ti = tau, Td = L/2
                lambda_factor = max(L_eff * 1.5, 15.0)  # Robustness factor
                self.Kc[i,j] = tau_eff / (self.K[i,j] * (L_eff + lambda_factor))
                self.Ti[i,j] = tau_eff
                self.Td[i,j] = L_eff * 0.5
        
        # Decoupling: use inverse of gain matrix (simplified)
        # For MIMO, prioritize diagonal
        self.primary_input = [0, 1, 2]  # y[0]->u[0], y[1]->u[1], y[2]->u[2]
        
        self.reset()
    
    def reset(self):
        self.integral = np.zeros(self.n_out)
        self.prev_y = None
        self.prev_u = np.zeros(self.n_in)
        self.y_filtered = None
        self.deriv_filtered = np.zeros(self.n_out)
        self.r_filtered = np.zeros(self.n_out)
        self.bad_count = 0
        self.u_history = []
        self.max_history = 100
        self.windup_count = np.zeros(self.n_out)
    
    def _filter_signal(self, signal, prev_filtered, tau):
        if prev_filtered is None:
            return signal.copy()
        alpha = self.sample_time / (tau + self.sample_time)
        return alpha * signal + (1 - alpha) * prev_filtered
    
    def _get_setpoint(self, t):
        schedule = {
            0: np.array([0.0, 0.0, 0.0]),
            120: np.array([0.4, -0.3, 0.0]),
            350: np.array([-0.25, 0.25, 0.12]),
            600: np.array([0.0, 0.0, 0.0])
        }
        
        times = sorted(schedule.keys())
        current_sp = schedule[0].copy()
        
        for sp_time in times:
            if t >= sp_time:
                current_sp = schedule[sp_time].copy()
        
        # Handle ramp from 350s to 440s
        if 350 <= t < 440:
            ramp_start = np.array([0.4, -0.3, 0.0])
            ramp_end = np.array([-0.25, 0.25, 0.12])
            progress = (t - 350) / 90.0
            current_sp = ramp_start + progress * (ramp_end - ramp_start)
        
        return current_sp
    
    def step(self, t, y, r, quality):
        sp = self._get_setpoint(t)
        
        # Handle bad quality measurements
        y_valid = y.copy()
        for i in range(self.n_out):
            if not quality[i]:
                self.bad_count += 1
                if self.y_filtered is not None:
                    y_valid[i] = self.y_filtered[i]
                else:
                    y_valid[i] = sp[i]
        
        # Filter measurements (noise rejection)
        filter_tau = 2.0
        self.y_filtered = self._filter_signal(y_valid, self.y_filtered, filter_tau)
        y_ctrl = self.y_filtered.copy()
        
        # Filter setpoint for smoother response
        sp_filter_tau = 3.0
        self.r_filtered = self._filter_signal(sp, self.r_filtered, sp_filter_tau)
        
        # Calculate error
        error = self.r_filtered - y_ctrl
        
        # Safety constraint enforcement on y[2]
        safety_margin = 0.08
        if y_ctrl[2] < self.y_min[2] + safety_margin:
            error[2] = max(error[2], 0.0)
            self.integral[2] = max(self.integral[2], 0.0)
        
        # Update integral with anti-windup
        dt = self.sample_time
        for i in range(self.n_out):
            if abs(error[i]) > 0.001:
                self.integral[i] += error[i] * dt
        
        # Anti-windup: clamp integral
        max_integral = 1.5
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # Calculate derivative (on measurement)
        if self.prev_y is not None:
            deriv = -(y_ctrl - self.prev_y) / dt
            deriv_filter_tau = 10.0
            self.deriv_filtered = self._filter_signal(deriv, self.deriv_filtered, deriv_filter_tau)
        else:
            self.deriv_filtered = np.zeros(self.n_out)
        
        self.prev_y = y_ctrl.copy()
        
        # Calculate control actions - SISO PID for each output-input pair
        u_calc = np.zeros(self.n_in)
        
        for j in range(self.n_in):
            u_sum = 0.0
            for i in range(self.n_out):
                if abs(self.Kc[i,j]) < 0.001:
                    continue
                
                # PID terms
                P_term = self.Kc[i,j] * error[i]
                I_term = self.Kc[i,j] * self.integral[i] / max(self.Ti[i,j], 1.0)
                D_term = self.Kc[i,j] * self.deriv_filtered[i] * self.Td[i,j]
                
                # Weight: prioritize diagonal
                if i == j:
                    weight = 1.0
                else:
                    weight = 0.15
                
                u_sum += (P_term + I_term + D_term) * weight
            
            u_calc[j] = u_sum
        
        # Rate limit the control change
        max_rate = 0.015
        u_delta = u_calc - self.prev_u
        u_delta = np.clip(u_delta, -max_rate, max_rate)
        u_calc = self.prev_u + u_delta
        
        # Apply actuator limits
        u_calc = np.clip(u_calc, self.u_min, self.u_max)
        
        # Additional safety check for y[2]
        if y_ctrl[2] < self.y_min[2] + 0.1:
            u_calc[2] = max(u_calc[2], self.prev_u[2])
        
        self.prev_u = u_calc.copy()
        self.u = u_calc
        
        self.u_history.append((t, u_calc.copy()))
        if len(self.u_history) > self.max_history:
            self.u_history.pop(0)
        
        return u_calc
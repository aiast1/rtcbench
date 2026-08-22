import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.Kp_level = 0.3  # Reduced gain for safety
        self.Ki_level = 0.015
        self.Kp_flow = 0.6
        self.Ki_flow = 0.04
        self.max_slew = 1.5  # Tighter rate limiting
        
        self.level_integral = 0.0
        self.flow_integral = 45.0/0.6  # Initial bias
        self.last_good_y = np.array([0.0, 85.0, 50.0, 45.0])
        self.l_inv_estimate = 0.0
        self.prev_valve = 45.0
        self.prev_error_level = 0.0

    def reset(self):
        self.level_integral = 0.0
        self.flow_integral = 45.0/0.6
        self.last_good_y = np.array([0.0, 85.0, 50.0, 45.0])
        self.l_inv_estimate = 0.0
        self.prev_valve = 45.0
        self.prev_error_level = 0.0

    def step(self, t, y, r, quality):
        # Update last good measurements with moving average
        for i in range(4):
            if quality[i]:
                self.last_good_y[i] = 0.8*self.last_good_y[i] + 0.2*y[i]
                
        l_ind, p, steam_flow, fw_flow = self.last_good_y
        setpoint = r[0] if not np.isnan(r[0]) else 0.0
        
        # Update inventory estimate with leakage
        self.l_inv_estimate = 0.95*self.l_inv_estimate + 0.05*((fw_flow - steam_flow) * self.sample_time / 20.0)
        
        # Safety override with deadband
        safety_bias = 0.0
        safe_max_ind = 230.0
        if l_ind > safe_max_ind:
            safety_bias -= 0.8 * (l_ind - safe_max_ind)
        elif self.l_inv_estimate < -200.0:
            safety_bias += 0.6 * (200.0 + self.l_inv_estimate)
        
        # Three-element controller with pressure compensation
        pressure_factor = np.clip((p - 70.0)/30.0, 0.5, 1.5)
        error_level = (setpoint - l_ind) * pressure_factor
        d_error = (error_level - self.prev_error_level)/self.sample_time
        level_output = self.Kp_level*(error_level + 0.3*d_error) + self.level_integral
        
        # Flow control with dynamic feedforward
        desired_flow = steam_flow + level_output + safety_bias
        error_flow = desired_flow - fw_flow
        raw_valve = self.Kp_flow*error_flow + self.flow_integral
        
        # Strict rate limiting
        valve = np.clip(raw_valve, 0.0, 100.0)
        valve = np.clip(valve, 
                      self.prev_valve - self.max_slew, 
                      self.prev_valve + self.max_slew)
        
        # Anti-windup with tracking
        if abs(raw_valve - valve) < 1e-3:
            self.level_integral += self.Ki_level * error_level * self.sample_time
            self.flow_integral += self.Ki_flow * error_flow * self.sample_time
        else:
            self.flow_integral = valve - self.Kp_flow*error_flow
            self.level_integral = level_output - self.Kp_level*error_level
            
        self.prev_valve = valve
        self.prev_error_level = error_level
        
        return np.array([valve])
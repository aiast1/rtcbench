import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = brief.sample_time
        
        # Ultra-conservative level controller - safety first
        self.Kp_level = 0.005   # Extremely low gain
        self.Ti_level = 800.0   # Very long integral time
        self.Td_level = 0.0
        
        # Flow controller
        self.Kp_flow = 0.5
        self.Ti_flow = 30.0
        
        # Valve limits
        self.valve_min = 0.0
        self.valve_max = 100.0
        
        # Rate limiting
        self.max_valve_rate = 0.5  # Very slow: 0.5% per 5s = 0.1%/s
        
        self.reset()
        
    def reset(self):
        # Integral states
        self.integral_level = 0.0
        self.integral_flow = 0.0
        
        # Previous values
        self.prev_level = 0.0
        self.prev_feedwater_flow = 18.78
        
        # Initial valve position
        self.valve_position = 45.0
        
        # Flow setpoint
        self.flow_setpoint = 18.78
        
        # Track steam flow for feedforward
        self.prev_steam_flow = 50.0  # Assume ~50 kg/s at steady state
        
        self.first_call = True
        
    def step(self, t, y, r, quality):
        # y[0] = LT-101 (level indicated) - SCORED
        # y[1] = PT-102 (pressure)
        # y[2] = FT-103 (steam flow)
        # y[3] = FT-104 (feedwater flow)
        
        level = y[0]
        pressure = y[1]
        steam_flow = y[2]
        feedwater_flow = y[3]
        
        # Get setpoint
        level_sp = r[0]
        
        # Handle bad measurements
        if not quality[0]:
            level = self.prev_level
        if not quality[3]:
            feedwater_flow = self.prev_feedwater_flow
        if not quality[2]:
            steam_flow = self.prev_steam_flow
        
        # === Feedforward from steam flow ===
        # When steam flow increases, we need more feedwater
        # This helps with the inverse response
        steam_deviation = steam_flow - self.prev_steam_flow
        feedforward_term = steam_deviation * 0.3  # Small feedforward
        
        # === Level Controller (PI) ===
        level_error = level_sp - level
        
        # Proportional term
        P_level = self.Kp_level * level_error
        
        # Integral term
        self.integral_level += level_error * self.Ts / self.Ti_level
        
        # Compute desired flow setpoint
        flow_output = P_level + self.integral_level + feedforward_term
        
        # Limit flow setpoint
        flow_setpoint_max = 40.0  # Conservative
        flow_setpoint_min = 0.0
        
        flow_setpoint = np.clip(flow_output, flow_setpoint_min, flow_setpoint_max)
        
        # Anti-windup
        if flow_output != flow_setpoint:
            saturation_error = flow_output - flow_setpoint
            self.integral_level -= saturation_error
        
        # === Flow Controller (PI) ===
        flow_error = flow_setpoint - feedwater_flow
        
        # Proportional term
        P_flow = self.Kp_flow * flow_error
        
        # Integral term
        self.integral_flow += flow_error * self.Ts / self.Ti_flow
        
        # Flow controller output
        valve_output = P_flow + self.integral_flow
        
        # Anti-windup for valve saturation
        if valve_output < self.valve_min:
            self.integral_flow -= (self.valve_min - valve_output) * 0.1
            valve_output = self.valve_min
        elif valve_output > self.valve_max:
            self.integral_flow -= (valve_output - self.valve_max) * 0.1
            valve_output = self.valve_max
        
        # Rate limiting
        valve_rate = valve_output - self.valve_position
        if abs(valve_rate) > self.max_valve_rate:
            valve_rate = np.sign(valve_rate) * self.max_valve_rate
        
        valve_position = self.valve_position + valve_rate
        
        # Hard limits
        valve_position = np.clip(valve_position, self.valve_min, self.valve_max)
        
        # Store for next iteration
        self.prev_level = level
        self.prev_feedwater_flow = feedwater_flow
        self.prev_steam_flow = steam_flow
        self.flow_setpoint = flow_setpoint
        self.valve_position = valve_position
        self.first_call = False
        
        return np.array([valve_position])
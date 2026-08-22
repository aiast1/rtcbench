import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        self.min_output = 270.0
        self.max_output = 340.0
        self.max_delta = 1.2 * self.dt  # Reduced rate limit
        self.duty_limit = 0.01715
        
        # Conservative PID parameters with safety margin
        self.Kp = 4.5
        self.Ki = 0.025
        self.Kd = 0.8
        self.tau_deriv = 8.0  # Tighter derivative filtering
        self.tau_filter = 15.0
        self.output_filter_tau = 6.0
        
        # Safety boundaries from brief
        self.T_min = 300.0
        self.T_max = 470.0
        
        self.reset()

    def reset(self):
        self.integral = 300.0 / self.Ki  # Bumpless start
        self.last_output = 300.0
        self.filtered_output = 300.0
        self.last_valid_pv = 350.0
        self.filtered_pv = 350.0
        self.deriv_filter = 0.0
        self.prev_pv = 350.0
        self.duty_accumulator = 0.0
        self.safety_margin = 0.0

    def step(self, t, y, r, quality):
        # Handle measurement quality
        if quality[0]:
            current_raw = y[0]
            self.last_valid_pv = current_raw
        else:
            current_raw = self.last_valid_pv
        
        # Safety-oriented filtering
        alpha_pv = self.dt / (self.tau_filter + self.dt)
        self.filtered_pv = alpha_pv * current_raw + (1 - alpha_pv) * self.filtered_pv
        pv = self.filtered_pv
        
        # Emergency safety override
        if pv > self.T_max - 15.0 or pv < self.T_min + 15.0:
            self.Ki = min(self.Ki * 1.2, 0.04)  # Boost integral near limits
        else:
            self.Ki = 0.025  # Reset to nominal
            
        # Conservative setpoint following
        setpoint = r[0] if not np.isnan(r[0]) else 350.0
        error = setpoint - pv
        
        # Derivative on PV only (avoid setpoint spikes)
        delta_pv = pv - self.prev_pv
        deriv_raw = delta_pv / self.dt
        alpha_deriv = self.dt / (self.tau_deriv + self.dt)
        self.deriv_filter = alpha_deriv * deriv_raw + (1 - alpha_deriv) * self.deriv_filter
        derivative = self.deriv_filter
        self.prev_pv = pv
        
        # PID calculation
        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * derivative
        output_unclamped = P + I + D
        
        # Aggressive anti-windup with clamping
        output_clamped = np.clip(output_unclamped, self.min_output, self.max_output)
        self.integral += (output_clamped - output_unclamped)/self.Ki if abs(output_clamped - output_unclamped) > 1e-3 else error * self.dt
        
        # Rate limiting with inertia
        delta = output_clamped - self.filtered_output
        delta_clamped = np.clip(delta, -self.max_delta, self.max_delta)
        new_output = self.filtered_output + delta_clamped
        
        # Output filtering with adaptive time constant
        alpha_out = self.dt / (self.output_filter_tau + self.dt)
        self.filtered_output = alpha_out * new_output + (1 - alpha_out) * self.filtered_output
        final_output = np.clip(self.filtered_output, self.min_output, self.max_output)
        
        # Predictive safety margin
        if pv > 400.0 or pv < 320.0:  # Approach safety limits
            final_output = np.clip(final_output, 280.0, 340.0)  # Conservative bounds
            
        # Duty management
        self.duty_accumulator += abs(final_output - self.last_output)/self.dt
        self.last_output = final_output
        
        # Emergency duty control
        if self.duty_accumulator > self.duty_limit * 0.8:
            self.filtered_output = final_output  # Bypass filter to reduce lag
            self.max_delta = 0.8 * self.dt  # Tighten rate limit
            
        return np.array([final_output])
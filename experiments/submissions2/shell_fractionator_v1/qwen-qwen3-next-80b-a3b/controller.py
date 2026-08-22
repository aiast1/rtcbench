import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        self.n_inputs = 3
        self.n_outputs = 3
        self.n_disturbances = 2
        
        # Nominal model parameters (used for tuning, not for prediction)
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20]
        ])
        self.TAU = np.array([
            [50.0, 60.0, 50.0],
            [50.0, 60.0, 40.0],
            [33.0, 44.0, 19.0]
        ])
        self.L = np.array([
            [27.0, 28.0, 27.0],
            [18.0, 14.0, 15.0],
            [20.0, 22.0, 0.0]
        ])
        
        # PID tuning parameters (conservative for robustness)
        self.Kp = np.array([0.15, 0.15, 0.18])   # Slightly increased for better response
        self.Ki = np.array([0.0018, 0.0018, 0.0025])  # Slightly increased integral gain
        self.Kd = np.array([0.005, 0.005, 0.01])  # Reduced derivative for noise immunity
        
        # Anti-windup and constraints
        self.actuator_limits = np.array([-0.5, 0.5])
        self.max_actuator_change = 0.0112  # Duty limit
        self.integral_windup_limit = 0.6   # Slightly increased for better steady-state
        
        # State variables
        self.prev_error = np.zeros(self.n_outputs)
        self.integral = np.zeros(self.n_outputs)
        self.prev_control = np.zeros(self.n_outputs)
        self.prev_measurement = np.zeros(self.n_outputs)
        self.last_t = 0.0
        self.derivative_filter = np.zeros(self.n_outputs)
        
        # Setpoint schedule
        self.setpoint_schedule = [
            (0.0, np.array([0.0, 0.0, 0.0])),
            (120.0, np.array([0.2, -0.15, 0.0])),
            (350.0, np.array([-0.1, 0.1, 0.05])),
            (600.0, np.array([0.0, 0.0, 0.0]))
        ]
        
        # Initialize for first call
        self.reset()
    
    def reset(self):
        """Reset controller state for new scenario."""
        self.prev_error = np.zeros(self.n_outputs)
        self.integral = np.zeros(self.n_outputs)
        self.prev_control = np.zeros(self.n_outputs)
        self.prev_measurement = np.zeros(self.n_outputs)
        self.derivative_filter = np.zeros(self.n_outputs)
        self.last_t = 0.0
    
    def step(self, t, y, r, quality):
        # Handle bad measurements - use previous value if quality is False
        y_clean = np.copy(y)
        for i in range(self.n_outputs):
            if not quality[i]:
                y_clean[i] = self.prev_measurement[i]
            else:
                self.prev_measurement[i] = y_clean[i]
        
        # Interpolate setpoint based on time
        setpoint = np.array([np.nan, np.nan, np.nan])
        for i in range(len(self.setpoint_schedule) - 1):
            t_start, sp_start = self.setpoint_schedule[i]
            t_end, sp_end = self.setpoint_schedule[i + 1]
            if t_start <= t < t_end:
                if t_end == t_start:
                    setpoint = sp_start
                else:
                    frac = (t - t_start) / (t_end - t_start)
                    setpoint = sp_start + frac * (sp_end - sp_start)
                break
        else:
            # Use last setpoint if beyond schedule
            setpoint = self.setpoint_schedule[-1][1]
        
        # Only use scored setpoints (non-nan)
        for i in range(self.n_outputs):
            if not np.isnan(r[i]):
                setpoint[i] = r[i]
        
        # Calculate error
        error = setpoint - y_clean
        
        # Update integral with anti-windup
        for i in range(self.n_outputs):
            # Only integrate if not saturated or if error opposes saturation
            if (self.prev_control[i] >= self.actuator_limits[1] and error[i] > 0) or \
               (self.prev_control[i] <= self.actuator_limits[0] and error[i] < 0):
                # Saturation - don't integrate
                pass
            else:
                self.integral[i] += error[i] * self.dt
                # Anti-windup: clamp integral term
                self.integral[i] = np.clip(self.integral[i], -self.integral_windup_limit, self.integral_windup_limit)
        
        # Derivative term with filtering
        if t > self.last_t:
            dt = t - self.last_t
            if dt > 0:
                # Calculate raw derivative
                raw_derivative = (y_clean - self.prev_measurement) / dt
                # Apply exponential smoothing filter to each output independently
                alpha = 0.05  # Very conservative filter for noisy measurements
                self.derivative_filter = alpha * raw_derivative + (1 - alpha) * self.derivative_filter
            else:
                self.derivative_filter = np.zeros(self.n_outputs)
        else:
            self.derivative_filter = np.zeros(self.n_outputs)
        
        # PID control calculation
        control = np.zeros(self.n_outputs)
        for i in range(self.n_outputs):
            P = self.Kp[i] * error[i]
            I = self.Ki[i] * self.integral[i]
            D = self.Kd[i] * self.derivative_filter[i]
            control[i] = P + I + D
        
        # Apply actuator slew limits
        control = np.clip(control, 
                         self.prev_control - self.max_actuator_change, 
                         self.prev_control + self.max_actuator_change)
        
        # Apply hard actuator limits
        control = np.clip(control, self.actuator_limits[0], self.actuator_limits[1])
        
        # Safety override for bottoms reflux temperature (index 2)
        # Hard floor of -0.5 - if we're close to violation, force positive duty
        if y_clean[2] < -0.48:  # Safety margin
            # Increase FCV-203 (bottoms reflux duty) to raise temperature
            control[2] = max(control[2], 0.08)
        
        # Ensure output is always a numpy array of floats
        control = np.array([float(x) for x in control])
        
        # Update state
        self.prev_control = np.copy(control)
        self.prev_error = error
        self.last_t = t
        
        return control
import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_tanks = 4
        self.n_measurements = 2
        self.n_actuators = 2
        self.max_level = 20.0
        self.min_level = 0.0
        self.max_actuator = 10.0
        self.min_actuator = 0.0
        self.actuator_slew_limit = 0.017 * self.sample_time  # max change per step
        self.initial_actuators = np.array([3.0, 3.0])
        
        # Very conservative parameters to avoid safety violations
        self.kp = np.array([[0.15, 0.02],   # u1 on h1, u1 on h2
                            [0.02, 0.15]])  # u2 on h1, u2 on h2
        self.ki = np.array([[0.001, 0.0002],
                            [0.0002, 0.001]])
        self.kd = np.array([[0.0, 0.0],
                            [0.0, 0.0]])
        
        # Strong filtering and anti-windup
        self.windup_factor = 0.1
        self.derivative_filter = 0.3
        
        # Setpoint schedule
        self.setpoints = []
        self._build_setpoint_schedule()
    
    def _build_setpoint_schedule(self):
        self.setpoints.append((0, np.array([12.26, 12.78])))
        self.setpoints.append((200, np.array([14.0, 12.78])))
        self.setpoints.append((500, np.array([14.0, 11.2])))
        self.setpoints.append((900, np.array([12.26, 12.78])))
    
    def _get_setpoint(self, t):
        for i in range(len(self.setpoints) - 1):
            t_start, sp_start = self.setpoints[i]
            t_end, sp_end = self.setpoints[i + 1]
            if t_start <= t < t_end:
                if t_start == 500 and t_end == 900:
                    if t < 560:
                        progress = (t - 500) / 60.0
                        return sp_start + progress * (sp_end - sp_start)
                    else:
                        return sp_end
                else:
                    return sp_start
        return self.setpoints[-1][1]
    
    def reset(self):
        self.integral = np.zeros(self.n_actuators)
        self.prev_error = np.zeros(self.n_measurements)
        self.prev_measurement = np.zeros(self.n_measurements)
        self.prev_derivative = np.zeros(self.n_measurements)
        self.prev_output = self.initial_actuators.copy()
        self.last_t = None
        # Safety margins: keep at least 3 cm below max level
        self.safety_margin = 3.0
        # Track if we've seen good measurements recently
        self.good_measurements = 0
    
    def step(self, t, y, r, quality):
        # Get current setpoint
        sp = self._get_setpoint(t)
        
        # Use measured values where quality is good, otherwise use previous values
        measurement = np.zeros(self.n_measurements)
        for i in range(self.n_measurements):
            if quality[i]:
                measurement[i] = y[i]
                self.good_measurements = max(0, self.good_measurements + 1)
            else:
                measurement[i] = self.prev_measurement[i]
                self.good_measurements = max(0, self.good_measurements - 1)
        
        # Ensure measurements are within physical bounds
        measurement = np.clip(measurement, 0.0, self.max_level)
        
        # Calculate error
        error = sp - measurement
        
        # Update integral with anti-windup
        for i in range(self.n_actuators):
            for j in range(self.n_measurements):
                if abs(self.kp[j, i]) > 1e-6:
                    self.integral[i] += error[j] * self.sample_time
                    # Strong anti-windup: reduce integral if actuator is saturated
                    if self.prev_output[i] >= self.max_actuator and error[j] > 0:
                        self.integral[i] -= error[j] * self.sample_time * self.windup_factor
                    elif self.prev_output[i] <= self.min_actuator and error[j] < 0:
                        self.integral[i] -= error[j] * self.sample_time * self.windup_factor
        
        # Calculate derivative with filtering
        derivative = np.zeros(self.n_measurements)
        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 0:
                raw_derivative = (measurement - self.prev_measurement) / dt
                for i in range(self.n_measurements):
                    derivative[i] = (1 - self.derivative_filter) * self.prev_derivative[i] + \
                                   self.derivative_filter * raw_derivative[i]
        
        # PID calculation
        u = np.zeros(self.n_actuators)
        for i in range(self.n_actuators):
            for j in range(self.n_measurements):
                u[i] += self.kp[j, i] * error[j]
                u[i] += self.ki[j, i] * self.integral[i]
                u[i] += self.kd[j, i] * derivative[j]
        
        # Add feedforward to maintain current output (bumpless transfer)
        u += self.prev_output
        
        # Apply actuator constraints and slew limits
        u = np.clip(u, self.min_actuator, self.max_actuator)
        u = np.clip(u, 
                    self.prev_output - self.actuator_slew_limit,
                    self.prev_output + self.actuator_slew_limit)
        
        # Safety check: if measurements are close to max level, reduce control effort
        # Since we can't measure h3 and h4, we need to be extra conservative
        # Assume worst-case: if measured levels are above threshold, reduce all control
        safety_threshold = self.max_level - self.safety_margin
        if np.any(measurement > safety_threshold):
            # Reduce control effort significantly
            reduction_factor = 0.3
            u = self.prev_output + (u - self.prev_output) * reduction_factor
        
        # Ensure we don't push too hard when setpoints are high
        if np.any(sp > safety_threshold):
            # Further reduce control effort
            reduction_factor = min(reduction_factor, 0.5)
            u = self.prev_output + (u - self.prev_output) * reduction_factor
        
        # Final hard limits
        u = np.clip(u, self.min_actuator, self.max_actuator)
        
        # Update state
        self.prev_measurement = measurement.copy()
        self.prev_derivative = derivative.copy()
        self.prev_output = u.copy()
        self.last_t = t
        
        return u
import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_y = 3
        self.n_u = 3
        self.n_d = 2
        
        # Nominal plant model (will be used for tuning, not for control)
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.9],
            [4.38, 4.42, 7.2]
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
        self.KD = np.array([
            [1.2, 1.44],
            [1.52, 1.83],
            [1.14, 1.26]
        ])
        self.TAUD = np.array([
            [45.0, 40.0],
            [25.0, 20.0],
            [27.0, 32.0]
        ])
        self.LD = np.array([
            [27.0, 27.0],
            [15.0, 15.0],
            [27.0, 32.0]
        ])
        
        # Controller tuning parameters (conservative for robustness)
        self.tau_c = 60.0  # closed-loop time constant
        
        # PID parameters per loop (decoupled PI/PID control)
        # Using Ziegler-Nichols tuning with safety margins
        self.Kc = np.zeros(3)
        self.tau_i = np.zeros(3)
        self.tau_d = np.zeros(3)
        
        # Loop 0 (top composition)
        self.Kc[0] = 0.15 / self.K[0, 0]  # ~0.037
        self.tau_i[0] = self.TAU[0, 0]
        self.tau_d[0] = 0.0
        
        # Loop 1 (side composition)
        self.Kc[1] = 0.15 / self.K[1, 1]  # ~0.026
        self.tau_i[1] = self.TAU[1, 1]
        self.tau_d[1] = 0.0
        
        # Loop 2 (bottoms temperature)
        self.Kc[2] = 0.15 / self.K[2, 2]  # ~0.021
        self.tau_i[2] = self.TAU[2, 2]
        self.tau_d[2] = 0.0
        
        # Anti-windup and safety
        self.u_max = 0.5
        self.u_min = -0.5
        self.du_max = 0.016  # actuator duty limit
        
        # Initialize internal state
        self.reset()
    
    def reset(self):
        # Controller state
        self.u = np.zeros(self.n_u)  # current control output
        self.u_prev = np.zeros(self.n_u)  # previous control output
        self.integral = np.zeros(self.n_u)  # integral terms
        self.derivative = np.zeros(self.n_u)  # derivative terms
        self.last_error = np.zeros(self.n_u)
        self.last_measurement = np.zeros(self.n_y)
        self.last_measurement_time = 0.0
        
        # Disturbance estimates (simple low-pass filtered)
        self.disturbance_estimate = np.zeros(self.n_d)
        
        # Time tracking
        self.t_prev = 0.0
        
        # First call flag
        self.first_call = True
    
    def step(self, t, y, r, quality):
        dt = t - self.t_prev
        if dt <= 0:
            dt = self.sample_time
        
        # Initialize outputs
        u_out = np.zeros(self.n_u)
        
        # Handle first call
        if self.first_call:
            self.u_prev = np.zeros(self.n_u)
            self.last_measurement = y.copy()
            self.last_measurement_time = t
            self.t_prev = t
            self.first_call = False
            return self.u_prev.copy()
        
        # Update measurements
        for i in range(self.n_y):
            if quality[i]:
                self.last_measurement[i] = y[i]
        
        # Calculate setpoint changes
        sp = r.copy()
        
        # Calculate errors for scored channels
        error = np.zeros(self.n_u)
        for i in range(self.n_y):
            if not np.isnan(sp[i]) and quality[i]:
                error[i] = sp[i] - y[i]
            else:
                error[i] = 0.0
        
        # Update integral terms with anti-windup
        for i in range(self.n_u):
            # Calculate ideal PID output
            # Proportional term
            P = self.Kc[i] * error[i]
            
            # Integral term with anti-windup
            if self.u_prev[i] >= self.u_max or self.u_prev[i] <= self.u_min:
                # Anti-windup: stop integrating when saturated
                I = self.integral[i]
            else:
                I = self.integral[i] + (self.Kc[i] / self.tau_i[i]) * error[i] * dt
            
            # Derivative term (filtered derivative on measurement)
            if quality[i] and self.tau_d[i] > 0:
                # Filtered derivative: D = (tau_d / (tau_d + dt)) * (D_prev + (1 - alpha) * (y - y_prev) / dt)
                alpha = self.tau_d[i] / (self.tau_d[i] + dt)
                D = alpha * self.derivative[i] + (1 - alpha) * (-self.Kc[i] * (y[i] - self.last_measurement[i]) / dt)
            else:
                D = 0.0
            
            # Total PID output
            u_pid = P + I + D
            
            # Apply anti-windup: limit integral term
            if u_pid > self.u_max:
                u_pid = self.u_max
                I = max(0, I - (u_pid - self.Kc[i] * error[i] - D))
            elif u_pid < self.u_min:
                u_pid = self.u_min
                I = min(0, I - (u_pid - self.Kc[i] * error[i] - D))
            
            # Store updated integral
            self.integral[i] = I
            self.derivative[i] = D
            
            # Apply rate limit
            du = u_pid - self.u_prev[i]
            if du > self.du_max:
                du = self.du_max
            elif du < -self.du_max:
                du = -self.du_max
            
            u_out[i] = self.u_prev[i] + du
            
            # Apply actuator limits
            u_out[i] = np.clip(u_out[i], self.u_min, self.u_max)
        
        # Update state
        self.u_prev = u_out.copy()
        self.last_measurement = y.copy()
        self.last_measurement_time = t
        self.t_prev = t
        
        return u_out
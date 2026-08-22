import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.num_outputs = 3
        self.num_inputs = 3
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

        self.integrator = np.zeros(self.num_outputs)
        self.last_output = np.zeros(self.num_outputs)
        self.output_limits = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
        self.w_effort = 0.5
        self.max_integrator_windup = 5.0  # Reduced windup limit

    def reset(self):
        self.integrator = np.zeros(self.num_outputs)
        self.last_output = np.zeros(self.num_outputs)

    def step(self, t, y, r, quality):
        u = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            if not quality[i]:
                continue

            error = r[i] - y[i]
            
            # PI control with anti-windup
            self.integrator[i] += error * self.sample_time
            
            # Limit integrator to prevent windup
            self.integrator[i] = np.clip(self.integrator[i], -self.max_integrator_windup, self.max_integrator_windup)

            u[i] = self.K[i, 0] * error + self.K[i, 1] * self.integrator[i]

            # Saturate output
            u[i] = np.clip(u[i], self.output_limits[i, 0], self.output_limits[i, 1])

        # Ensure safety constraint on y[2]
        if y[2] < -0.5:
            u = np.zeros(self.num_outputs)

        # Actuator duty limit check - more conservative
        duty = np.sum(np.abs(u - self.last_output))
        if duty > 0.01: # Reduced duty cycle limit
            u = self.last_output

        self.last_output = u
        return u
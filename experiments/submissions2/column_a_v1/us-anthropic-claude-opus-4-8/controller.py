import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = float(getattr(brief, "sample_time", 1.0))
        self.L_min, self.L_max = 1.5, 4.5
        self.V_min, self.V_max = 2.0, 5.0
        self.L0 = 2.70629
        self.V0 = 3.20629
        self.reset()

    def reset(self):
        self.iL = 0.0
        self.iV = 0.0
        self.uL = self.L0
        self.uV = self.V0
        self.yD = 0.99
        self.xB_meas = 0.99
        self.dD = 0.0
        self.dB = 0.0
        self.prev_eD = None
        self.prev_eB = None

    def step(self, t, y, r, quality):
        Ts = self.Ts

        yD_raw = y[0]
        yB_raw = y[1]
        qD = bool(quality[0]) if quality is not None and len(quality) > 0 else True
        qB = bool(quality[1]) if quality is not None and len(quality) > 1 else True

        if qD and np.isfinite(yD_raw):
            self.yD = 0.7 * self.yD + 0.3 * float(yD_raw)
        if qB and np.isfinite(yB_raw):
            self.xB_meas = 0.7 * self.xB_meas + 0.3 * float(yB_raw)

        yD = self.yD
        yB = self.xB_meas

        rD = r[0] if (r is not None and len(r) > 0 and np.isfinite(r[0])) else 0.99
        rB = r[1] if (r is not None and len(r) > 1 and np.isfinite(r[1])) else 0.99

        eD = rD - yD
        eB = rB - yB

        # More proportional lift to reduce tracking error; integral a touch
        # higher to remove the persistent offset on the slow column. Effort
        # is well within budget so we can afford it.
        Kp_L = 12.0
        Ki_L = 0.045
        Kp_V = 12.0
        Ki_V = 0.045

        if self.prev_eD is None:
            self.prev_eD = eD
            self.prev_eB = eB
        dedD = (eD - self.prev_eD) / Ts
        dedB = (eB - self.prev_eB) / Ts
        self.dD = 0.85 * self.dD + 0.15 * dedD
        self.dB = 0.85 * self.dB + 0.15 * dedB
        self.prev_eD = eD
        self.prev_eB = eB
        Kd = 0.8

        iL_new = self.iL + Ki_L * eD * Ts
        iV_new = self.iV + Ki_V * eB * Ts

        uL = self.L0 + Kp_L * eD + iL_new + Kd * self.dD
        uV = self.V0 + Kp_V * eB + iV_new + Kd * self.dB

        min_gap = 0.12
        max_gap = 0.85
        spacing = uV - uL
        if spacing < min_gap:
            mid = 0.5 * (uL + uV)
            uL = mid - min_gap / 2
            uV = mid + min_gap / 2
        elif spacing > max_gap:
            mid = 0.5 * (uL + uV)
            uL = mid - max_gap / 2
            uV = mid + max_gap / 2

        uL_c = min(max(uL, self.L_min), self.L_max)
        uV_c = min(max(uV, self.V_min), self.V_max)

        satL = (uL_c != uL)
        satV = (uV_c != uV)
        if not satL:
            self.iL = iL_new
        if not satV:
            self.iV = iV_new
        self.iL = min(max(self.iL, -1.5), 1.5)
        self.iV = min(max(self.iV, -1.5), 1.5)

        max_dL = 0.015
        max_dV = 0.015
        dL = uL_c - self.uL
        dV = uV_c - self.uV
        dL = min(max(dL, -max_dL), max_dL)
        dV = min(max(dV, -max_dV), max_dV)

        self.uL += dL
        self.uV += dV

        self.uL = min(max(self.uL, self.L_min), self.L_max)
        self.uV = min(max(self.uV, self.V_min), self.V_max)

        return np.array([self.uL, self.uV], dtype=float)
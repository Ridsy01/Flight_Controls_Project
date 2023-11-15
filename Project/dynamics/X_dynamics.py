import numpy as np
import Project.parameters.X_parameters as P


class XDynamics:
    def __init__(self, Ts, state0):
        # Initial state conditions
        self.state = state0
        # simulate time
        self.ts_simulation = Ts
        # jx
        self.jx = P.jx
        # jy
        self.jy = P.jy
        # jz
        self.jz = P.jz
        # jxz
        self.jxz = P.jxz
        # mass
        self.mass = P.mass
        # gravity constant
        self.g = P.g

    def update(self, f_m):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        fx = f_m.item(0)
        fy = f_m.item(1)
        fz = f_m.item(2)
        l = f_m.item(3)
        m = f_m.item(4)
        n = f_m.item(5)

        time_step = self.ts_simulation

        k1 = self.f(self.state, fx, fy, fz, l, m, n)
        k2 = self.f(self.state + time_step / 2. * k1, fx, fy, fz, l, m, n)
        k3 = self.f(self.state + time_step / 2. * k2, fx, fy, fz, l, m, n)
        k4 = self.f(self.state + time_step * k3, fx, fy, fz, l, m, n)

        self.state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # return the corresponding output

    def f(self, state, fx, fy, fz, l, m, n):
        # Return state values
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        phi = state.item(6)
        theta = state.item(7)
        psi = state.item(8)
        # e3 = state.item(9)
        p = state.item(9)
        q = state.item(10)
        r = state.item(11)

        # Rotational Dynamics Values
        ga = (self.jx * self.jz) - self.jxz ** 2
        ga1 = (self.jxz * (self.jx - self.jy + self.jz)) / ga
        ga2 = (self.jz * (self.jz - self.jy) + self.jz ** 2) / ga
        ga3 = self.jz / ga
        ga4 = self.jxz / ga
        ga5 = (self.jz - self.jx) / self.jy
        ga6 = self.jxz / self.jy
        ga7 = ((self.jx - self.jy) * self.jx + self.jxz ** 2) / ga
        ga8 = self.jx / ga

        # Matrix
        # Translational Kinematics
        Rbvt = np.array(
            ([[np.cos(theta) * np.cos(psi), (np.sin(phi) * np.sin(theta) * np.cos(psi)) - (np.cos(phi) * np.sin(psi)),
               (np.cos(phi) * np.sin(theta) * np.cos(psi)) - (np.sin(phi) * np.sin(psi))],
              [np.cos(theta) * np.sin(psi), (np.sin(phi) * np.sin(theta) * np.sin(psi)) + (np.cos(phi) * np.cos(psi)),
               (np.cos(phi) * np.sin(theta) * np.sin(psi)) - (np.sin(phi) * np.cos(psi))],
              [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]]))
        p_1 = np.array(([u], [v], [w]))

        # Rotational Kinematics
        Rbvr = np.array(([1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                         [0, np.cos(phi), -np.sin(phi)],
                         [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]))

        ro = np.array(([p], [q], [r]))

        # Translational Dynamics
        dif = np.array(([(r * v) - (q * w)], [(p * w) - (r * u)], [(q * u) - (p * v)]))
        fm = np.array(([fx / P.mass], [fy / P.mass], [fz / P.mass]))

        # Rotational Dynamics
        l1 = np.array(([(ga1 * p * q) - (ga2 * q * r)],
                       [(ga5 * p * r) - (ga6 * (p ** 2 - r ** 2))],
                       [(ga7 * p * q) - (ga1 * q * r)]))
        r1 = np.array(([(ga3 * l) + (ga4 * n)],
                       [(1 / P.jy) * m],
                       [(ga4 * l) + (ga8 * n)]))

        # Equations
        # Translational Kinematics
        Pdot = Rbvt @ p_1

        # Rotational Kinematics
        rotkdot = Rbvr @ ro

        # Translational Dynamics
        Vdot = dif + fm

        # Rotational Dynamics
        rotd_dot = l1 + r1

        # build xdot and return
        xdot = np.array(
            [[Pdot[0][0]], [Pdot[1][0]], [Pdot[2][0]], [Vdot[0][0]], [Vdot[1][0]], [Vdot[2][0]],
             [rotkdot[0][0]], [rotkdot[1][0]], [rotkdot[2][0]], [rotd_dot[0][0]], [rotd_dot[1][0]], [rotd_dot[2][0]]])
        return xdot


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

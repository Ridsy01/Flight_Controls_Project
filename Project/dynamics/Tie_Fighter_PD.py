import numpy as np
import Project.parameters.X_parameters as P


class PDctrltie:
    def __init__(self):
        zeta = .707
        tr = 0.1
        wn = 2.2 / tr

        deno = P.mass * P.g

        self.kdz = 2 * zeta * (wn) * (deno)
        self.kpz = 2*(wn**2) * (deno)
        print(self.kdz)
        print(self.kpz)

        deno = P.mass

        zeta = .0001
        tr = 1.
        wn = 2.2 / tr

        self.kdy = 2 * zeta * (wn) * deno
        self.kpy = ((2.2 / tr) ** 2) * deno
        print(self.kdy)
        print(self.kpy)

    def update(self, pd, pd_ref,pe, pe_ref, vdot, wdot):

        F_e_z = (P.mass*P.g)

        F_tilde_z = (self.kpz * -(pd - pd_ref) - self.kdz * wdot)
        F_tilde_y = -(self.kpy * (pe - pe_ref) - self.kdy * vdot + (30/(pe-pe_ref))*self.kpy)

        F_z = F_e_z + F_tilde_z
        F_y = 0. + F_tilde_y

        F_z = saturate(F_z, P.F_max)
        F_y = saturate(F_y, P.F_max)

        return F_z, F_y


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

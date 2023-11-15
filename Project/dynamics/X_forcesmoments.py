import numpy as np
import Project.parameters.X_parameters as P
import control
from control.matlab import *

def rotation_matrix_body2inertial(phi, theta, psi):
    r_b_w = np.array([[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)
                       + np.sin(phi)*np.sin(psi)],
                      [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)
                       - np.sin(phi) * np.cos(psi)],
                      [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])
    return r_b_w


def Euler2Rotation(phi, theta, psi):
    """
    Converts euler angles to rotation matrix (R_b^i)
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    #R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    R = R_yaw @ R_pitch @ R_roll

    # rotation is body to inertial frame
    # R = np.array([[c_theta*c_psi, s_phi*s_theta*c_psi-c_phi*s_psi, c_phi*s_theta*c_psi+s_phi*s_psi],
    #               [c_theta*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, c_phi*s_theta*s_psi-s_phi*c_psi],
    #               [-s_theta, s_phi*c_theta, c_phi*c_theta]])

    return R


class forcemoment:
    def __init__(self, alpha=0.0):
        self.Ts = P.Ts
        self.mass = P.mass
        self.g = P.g
        self.rho = P.rho
        self.e = P.e
        self.M = P.M
        self.alpha0 = P.alpha0
        self.S_wing = P.S_wing
        self.b = P.b
        self.c = P.c
        self.S_prop = P.S_prop
        self.AR = P.AR
        self.C_L_0 = P.C_L_0
        self.C_D_0 = P.C_D_0
        self.C_m_0 = P.C_m_0
        self.C_L_alpha = P.C_L_alpha
        self.C_D_alpha = P.C_D_alpha
        self.C_m_alpha = P.C_m_alpha
        self.C_L_q = P.C_L_q
        self.C_D_q = P.C_D_q
        self.C_m_q = P.C_m_q
        self.C_L_delta_e = P.C_L_delta_e
        self.C_D_delta_e = P.C_D_delta_e
        self.C_m_delta_e = P.C_m_delta_e
        self.M = P.M
        self.epsilon = P.epsilon
        self.C_D_p = P.C_D_p
        self.C_Y_0 = P.C_Y_0
        self.C_ell_0 = P.C_ell_0
        self.C_n_0 = P.C_n_0
        self.C_Y_beta = P.C_Y_beta
        self.C_ell_beta = P.C_ell_beta
        self.C_n_beta = P.C_n_beta
        self.C_Y_p = P.C_Y_p
        self.C_ell_p = P.C_ell_p  # ell=p
        self.C_n_p = P.C_n_p
        self.C_Y_r = P.C_Y_r
        self.C_ell_r = P.C_ell_r
        self.C_n_r = P.C_n_r
        self.C_Y_delta_a = P.C_Y_delta_a
        self.C_ell_delta_a = P.C_ell_delta_a
        self.C_n_delta_a = P.C_n_delta_a
        self.C_Y_delta_r = P.C_Y_delta_r
        self.C_ell_delta_r = P.C_ell_delta_r
        self.C_n_delta_r = P.C_n_delta_r
        self.C_prop = P.C_prop
        self.k_motor = P.k_motor
        self.k_tp = P.k_tp
        self.k_omega = P.k_omega
        self.Va0 = P.Va0
        self.wn = P.wn
        self.we = P.we
        self.wd = P.wd

    def wind(self, phi, theta, psi, Va, dt):

        Lu = 200
        Lv = 200
        Lw = 50
        sigma_u = 1.06
        sigma_v = sigma_u
        sigma_w = 0.7

        au = sigma_u * np.sqrt(2 * Va / Lu)
        av = sigma_v * np.sqrt(3 * Va / Lv)
        aw = sigma_w * np.sqrt(3 * Va / Lw)

        unum = [0, au]
        udenum = [1, Va / Lu]
        H_u = tf(unum, udenum)

        vnum = [av, av * Va / (np.sqrt(3) * Lv)]
        vdenum = [1, 2 * Va / Lv, (Va / Lv) ** 2]
        H_v = tf(vnum, vdenum)

        wnum = [aw, aw * Va / (np.sqrt(3) * Lw)]
        wdenum = [1, 2 * Va / Lw, (Va / Lw) ** 2]
        H_w = tf(wnum, wdenum)

        wn_u = np.random.normal(0, 1, 1)
        wn_v = np.random.normal(0, 1, 1)
        wn_w = np.random.normal(0, 1, 1)

        T = [0, dt]

        y_u, T, x_u = lsim(H_u, wn_u[0], T, 0.0)
        y_v, T, x_v = lsim(H_v, wn_v[0], T, 0.0)
        y_w, T, x_w = lsim(H_w, wn_w[0], T, 0.0)

        wg_u = y_u[1]
        wg_v = y_v[1]
        wg_w = y_w[1]
        wsv = np.array([self.wn, self.we, self.wd])
        R = Euler2Rotation(phi, theta, psi)
        wsb = np.matmul(R.T, wsv.T).T
        wgb = np.array([wg_u, wg_v, wg_w])
        vw = wgb + wsb
        return vw

    def air(self, states, Va0, dt):
        u = states.item(3)
        v = states.item(4)
        w = states.item(5)
        phi = states.item(6)
        theta = states.item(7)
        psi = states.item(8)

        vw = 0.0*self.wind(phi, theta, psi, Va0, dt)
        vab = np.array([u - vw[0], v - vw[1], w - vw[2]])
        va = np.sqrt(vab[0] ** 2 + vab[1] ** 2 + vab[2] ** 2)
        alpha = np.arctan2(vab[2], vab[0])
        # beta = np.arcsin(vab[1] / np.sqrt(vab[0] ** 2 + vab[1] ** 2 + vab[2] ** 2))
        beta = np.arctan2(vab[1], va)

        return va, alpha, beta, vw

    def f_m(self, states, delta):
        pn = states.item(0)
        pe = states.item(1)
        pd = states.item(2)
        u = states.item(3)
        v = states.item(4)
        w = states.item(5)
        phi = states.item(6)
        theta = states.item(7)
        psi = states.item(8)
        p = states.item(9)
        q = states.item(10)
        r = states.item(11)
        Va0 = self.Va0

        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        va, alpha, beta, vw = self.air(states, Va0, self.Ts)

        dynam = 0.5 * self.rho * va ** 2
        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        # Gravity Forces
        fx = -self.mass*self.g*np.sin(theta)
        fy = self.mass*self.g*np.cos(theta)*np.sin(phi)
        fz = self.mass*self.g*np.cos(theta)*np.cos(phi)

        # Lift and Drag
        tmp1 = np.exp(-self.M * (alpha - self.alpha0))
        tmp2 = np.exp(self.M * (alpha + self.alpha0))
        sigma = (1 + tmp1 + tmp2) / ((1 + tmp1) * (1 + tmp2))
        CL = (1 - sigma) * (self.C_L_0 + self.C_L_alpha * alpha)
        CD = self.C_D_p + 1 / (np.pi*self.e*self.AR)*(self.C_L_0+self.C_L_alpha*alpha)**2
        CL = CL + np.sin(alpha)*sigma*2*sina*sina*cosa

        # Aerodynamic Forces
        fx = fx + dynam*self.S_wing*(-CD*cosa + CL*sina)
        fx = fx + dynam*self.S_wing*(-self.C_D_q*cosa + self.C_L_q*sina)*self.c*q/(2*va)
        fy = fy + dynam*self.S_wing*(self.C_Y_0 + self.C_Y_beta*beta)
        fy = fy + dynam*self.S_wing*(self.C_Y_p*p + self.C_Y_r*r)*self.b/(2*va)
        fz = fz + dynam*self.S_wing*(-CD*sina - CL*cosa)
        fz = fz + dynam*self.S_wing*(-self.C_D_q*sina - self.C_L_q*cosa)*self.c*q/(2*va)

        # Aerodynamic Torques
        l = dynam*self.S_wing*self.b*(self.C_ell_0 + self.C_ell_beta*beta)
        l = l + dynam*self.S_wing*self.b*(self.C_ell_p*p + self.C_ell_r*r)*self.b/(2*va)
        m = dynam*self.S_wing*self.c*(self.C_m_0 + self.C_m_alpha*alpha)
        m = m + dynam*self.S_wing*self.c*self.C_m_q*self.c*q/(2*va)
        n = dynam*self.S_wing*self.b*(self.C_n_0 + self.C_n_beta*beta)
        n = n + dynam*self.S_wing*self.b*(self.C_n_p*p + self.C_n_r*r)*self.b/(2*va)

        # Control Forces
        fx = fx + dynam*self.S_wing*(-self.C_D_delta_e*cosa + self.C_L_delta_e*sina)*delta_e
        fy = fy + dynam*self.S_wing*(self.C_Y_delta_a*delta_a + self.C_Y_delta_r*delta_r)
        fz = fz + dynam*self.S_wing*(-self.C_D_delta_e*sina - self.C_L_delta_e*cosa)*delta_e

        # Control Torques
        l = l + dynam*self.S_wing*self.b*(self.C_ell_delta_a*delta_a + self.C_ell_delta_r*delta_r)
        m = m + dynam*self.S_wing*self.c*self.C_m_delta_e*delta_e
        n = n + dynam*self.S_wing*self.b*(self.C_n_delta_a*delta_a + self.C_n_delta_r*delta_r)

        # Propulsion Forces
        mt = ((self.k_motor*delta_t)**2)-va**2
        fx = fx + 0.5*self.rho*self.S_prop*self.C_prop*mt

        # Propulsion Torques
        tp = -self.k_tp*(self.k_omega*delta_t)**2
        l = l + tp

        f_m = np.array([[fx], [fy], [fz], [l], [m], [n]], dtype=float)

        return f_m

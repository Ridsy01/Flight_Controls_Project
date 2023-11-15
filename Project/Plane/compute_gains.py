import numpy as np
from Assignment1.Plane.compute_trim import ComputeTrim
import Assignment1.Plane.aerosonde_parameters as P
import control
from control.matlab import *


class computegains:
    def __init__(self):
        self.P = P
        self.computetrim = ComputeTrim()

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

        self.jx = P.jx
        self.jy = P.jy
        self.jz = P.jz
        self.jxz = P.jxz
        self.gamma = P.gamma
        self.gamma1 = P.gamma1
        self.gamma2 = P.gamma2
        self.gamma3 = P.gamma3
        self.gamma4 = P.gamma4
        self.gamma5 = P.gamma5
        self.gamma6 = P.gamma6
        self.gamma7 = P.gamma7
        self.gamma8 = P.gamma8

        # Cs
        self.cp0 = P.cp0
        self.cpb = P.cpb
        self.cpp = P.cpp
        self.cpr = P.cpr
        self.cpda = P.cpda
        self.cpdr = P.cpdr
        self.cr0 = P.cr0
        self.crb = P.crb
        self.crp = P.crp
        self.crr = P.crr
        self.crda = P.crda
        self.crdr = P.crdr

    def update(self, states, delta):
        t_phi_da, t_chi_phi, t_beta_dr, t_theta_de, t_h_theta, t_h_va, t_va_dt, t_va_theta = self.transfer_fun(states, delta)
        evalue_lat, evalue_lon = self.statespace(states, delta)
        self.modes(states, delta, evalue_lat, evalue_lon)

        return

    def transfer_fun(self, states, delta):
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

        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        va = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan(w / u)
        beta = np.arctan2(v, va)

        # Roll Constants
        a_phi1 = -0.5*self.rho*(va**2)*self.S_wing*self.b*self.cpp*(self.b/(2*va))
        a_phi2 = 0.5*self.rho*(va**2)*self.S_wing*self.b*self.cpda
        d_phi1 = q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        d_phi2 = 0.0

        # Course and Heading Constants
        d_X = np.tan(phi) - phi

        # Sideslip Constants
        a_beta1 = -(self.rho*va*self.S_wing*self.C_Y_beta)/(2*self.mass*np.cos(beta))
        a_beta2 = (self.rho*va*self.S_wing*self.C_Y_delta_r)/(2*self.mass*np.cos(beta))
        d_beta = (1/(va*np.cos(beta)))*(p*w - r*u + self.g*np.cos(theta)*np.sin(phi)) + (self.rho*va*self.S_wing/(2*self.mass*np.cos(beta)))*(self.C_Y_0 + self.C_Y_p*(self.b*p/(2*va)) + self.C_Y_r*(self.b*r/(2*va)) + self.C_Y_delta_a*delta_a)

        # Pitch Constants
        a_theta1 = -(self.rho*(va**2)*(self.c**2)*self.S_wing*self.C_m_q) / (2*self.jy*2*va)
        a_theta2 = -(self.rho*(va**2)*self.c*self.S_wing*self.C_m_alpha) / (2*self.jy)
        a_theta3 = (self.rho * (va ** 2) * self.c * self.S_wing * self.C_m_delta_e) / (2 * self.jy)
        d_theta1 = q*(np.cos(phi)-1)
        d_theta2 = 0.0

        # Altitude Constants
        d_h = (u*np.sin(theta) - va*theta) - v*np.sin(phi)*np.cos(theta) - w*np.cos(phi)*np.cos(theta)

        # Airspeed Constants
        a_v1 = (self.rho*va*self.S_wing / self.mass)*(self.C_D_0 + self.C_D_alpha*alpha + self.C_D_delta_e*delta_e) + (self.rho*self.S_prop / self.mass)*self.C_prop*va
        a_v2 = (self.rho*self.S_prop / self.mass)*self.C_prop*(self.k_motor**2)*delta_t
        a_v3 = self.g

        # Transfer functions
        t_phi_da = tf([a_phi2], [1, a_phi1, 0])
        t_chi_phi = tf([self.g/va], [1, 0])
        t_beta_dr = tf([a_beta2], [1, a_beta1])
        t_theta_de = tf([a_theta3], [1, a_theta1, a_theta2])
        t_h_theta = tf([va], [1, 0])
        t_h_va = tf([theta], [1, 0])
        t_va_dt = tf([a_v2], [1, a_v1])
        t_va_theta = tf([-a_v3], [1, a_v1])

        # LATERAL
        zeta_phi = 0.707
        # Roll
        t_phi = 0.1
        wn_phi = 2.2/t_phi
        kp_phi = (wn_phi**2)/a_phi2
        kd_phi = (2*zeta_phi*wn_phi - a_phi1) / a_phi2

        # Course Hold Loop
        zeta_x = 0.5
        Wx = 80.0
        wn_x = (1/Wx)*wn_phi
        kp_x = (2*zeta_x*wn_x*va) / self.g
        ki_x = (wn_x**2)*va/self.g

        # LONGITUDINAL
        # Pitch Attitude Hold
        zeta_theta = 0.1
        t_theta = 0.1
        wn_theta = 2.2/t_theta
        kp_theta = ((wn_theta**2) - a_theta2) / a_theta3
        kd_theta = (2*zeta_theta*wn_theta - a_theta1) / a_theta3
        DC_theta = (kp_theta*a_theta3) / (a_theta2 + kp_theta*a_theta3)

        # Altitude from Pitch
        Wh = 10.0
        zeta_h = 0.1
        wn_h = (1/Wh)*wn_theta
        kp_h = (2*zeta_h*wn_h) / (DC_theta*va)
        ki_h = (wn_h**2) / (DC_theta*va)

        # Airspeed from Pitch
        Wv2 = 1.1
        zeta_v2 = 0.707
        wn_v2 = (1/Wv2)*wn_theta
        kp_v2 = (a_v1-2*zeta_v2*wn_v2) / (DC_theta*self.g)
        ki_v2 = (wn_v2**2) / (DC_theta*self.g)

        # Airspeed from Throttle
        zeta_v = 0.707
        t_v = 0.5
        wn_v = 2.2/t_v
        kp_v = (2*zeta_v*wn_v - a_v1) / a_v2
        ki_v = (wn_v**2) / a_v2

        k_values = np.array([[kp_phi], [kd_phi], [kp_x], [ki_x], [kp_theta], [kd_theta], [kp_h], [ki_h], [kp_v2], [ki_v2], [kp_v], [ki_v]])
        return k_values

    def statespace(self, states, delta):
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

        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        va = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        alpha = np.arctan(w / u)
        beta = np.arctan2(v, va)

        # Lateral
        Yv = (self.rho*self.S_wing*self.b*v / (4*self.mass*va))*(self.C_Y_p*p + self.C_Y_r*r) + (self.rho*self.S_wing*v / self.mass)*(self.C_Y_0 + self.C_Y_beta*beta + self.C_Y_delta_a*delta_a + self.C_Y_delta_r*delta_r) + (self.rho*self.S_wing*self.C_Y_beta / (2*self.mass))*np.sqrt((u**2) + (w**2))
        Yp = w + (self.rho*va*self.S_wing*self.b*self.C_Y_p / (4*self.mass))
        Yr = -u + (self.rho*va*self.S_wing*self.b*self.C_Y_r / (4*self.mass))
        Yda = (self.rho*(va**2)*self.S_wing*self.C_Y_delta_a / (2*self.mass))
        Ydr = (self.rho*(va**2)*self.S_wing*self.C_Y_delta_r / (2*self.mass))
        Lv = (self.rho*self.S_wing*(self.b**2)*v / (4*va))*(self.cpp*p + self.cpr*r) + self.rho*self.S_wing*self.b*v*(self.cp0 + self.cpb*beta + self.cpda*delta_a + self.cpdr*delta_r) + (self.rho*self.S_wing*self.b*self.cpb / 2)*np.sqrt((u**2) + (w**2))
        Lp = self.gamma1*q + (self.rho*va*self.S_wing*(self.b**2)*self.cpp/4)
        Lr = -self.gamma2*q + (self.rho*va*self.S_wing*(self.b**2)*self.cpr/4)
        Lda = self.rho*(va**2)*self.S_wing*self.b*self.cpda / 2
        Ldr = self.rho*(va**2)*self.S_wing*self.b*self.cpdr/2
        Nv = (self.rho*self.S_wing*(self.b**2)*v / (4*va))*(self.crp*p + self.crr*r) + self.rho*self.S_wing*self.b*v*(self.cr0 + self.crb*beta + self.crda*delta_a + self.crdr*delta_r) + (self.rho*self.S_wing*self.b*self.crb / 2)*np.sqrt((u**2) + (w**2))
        Np = self.gamma7*q + (self.rho*va*self.S_wing*(self.b**2)*self.crp/4)
        Nr = -self.gamma1*q + (self.rho*va*self.S_wing*(self.b**2)*self.crr/4)
        Nda = self.rho*(va**2)*self.S_wing*self.b*self.crda / 2
        Ndr = self.rho*(va**2)*self.S_wing*self.b*self.crdr/2

        # Longitudinal
        C_L = self.C_L_0 + self.C_L_alpha * alpha
        C_D = self.C_D_0 + self.C_D_alpha * alpha

        C_X = -C_D * np.cos(alpha) + C_L * np.sin(alpha)
        C_X_q = -self.C_D_q * np.cos(alpha) + self.C_L_q * np.sin(alpha)
        C_X_delta_e = -self.C_D_delta_e * np.cos(alpha) + self.C_L_delta_e * np.sin(alpha)
        C_X_0 = -self.C_D_0 * np.cos(alpha) + self.C_L_0 * np.sin(alpha)
        C_X_alpha = -self.C_D_alpha * np.cos(alpha) + self.C_L_alpha * np.sin(alpha)

        C_Z = -C_D * np.sin(alpha) - C_L * np.cos(alpha)
        C_Z_q = -self.C_D_q * np.sin(alpha) - self.C_L_q * np.cos(alpha)
        C_Z_delta_e = -self.C_D_delta_e * np.sin(alpha) - self.C_L_delta_e * np.cos(alpha)
        C_Z_0 = -self.C_D_0 * np.sin(alpha) - self.C_L_0 * np.cos(alpha)
        C_Z_alpha = -self.C_D_alpha * np.sin(alpha) - self.C_L_alpha * np.cos(alpha)

        Xu = (u * self.rho * self.S_wing / self.mass) * (C_X_0 + C_X_alpha * alpha + C_X_delta_e * delta_e) - (self.rho * self.S_wing * w * C_X_alpha / (2 * self.mass)) + (self.rho * self.S_wing * self.c * C_X_q * u * q / (4 * self.mass * va)) - (self.rho * self.S_prop * self.C_prop * u / self.mass)
        Xw = -q + (w * self.rho * self.S_wing / self.mass) * (C_X_0 + C_X_alpha * alpha + C_X_delta_e * delta_e) + (self.rho * self.S_wing * self.c * C_X_q * w * q / (4 * self.mass * va)) + (self.rho * self.S_wing * u * C_X_alpha / (2 * self.mass)) - (self.rho * self.S_prop * self.C_prop * w / self.mass)
        Xq = -w + (self.rho * va * self.S_wing * C_X_q * self.c / (4 * self.mass))
        Xde = (self.rho * (va ** 2) * self.S_wing * C_X_delta_e / (2 * self.mass))
        Xdt = (self.rho * self.S_prop * self.C_prop * (self.k_motor ** 2) * delta_t) / self.mass
        Zu = q + (u * self.rho * self.S_wing / self.mass) * (C_Z_0 + C_Z_alpha * alpha + C_Z_delta_e * delta_e) - (self.rho * self.S_wing * w * C_Z_alpha / (2 * self.mass)) + (u*self.rho * self.S_wing * C_Z_q * self.c * q / (4 * self.mass * va))
        Zw = (w * self.rho * self.S_wing / self.mass) * (C_Z_0 + C_Z_alpha * alpha + C_Z_delta_e * delta_e) + (self.rho * self.S_wing * u * C_Z_alpha / (2 * self.mass)) + (self.rho*w*self.S_wing*self.c*C_Z_q*q/(4*self.mass*va))
        Zq = u + (self.rho*va*self.S_wing*C_Z_q*self.c / (4*self.mass))
        Zde = (self.rho*(va**2)*self.S_wing*C_Z_delta_e / (2*self.mass))
        Mu = (u*self.rho*self.S_wing*self.c / self.jy)*(self.C_m_0 + self.C_m_alpha*alpha + self.C_m_delta_e*delta_e) - (self.rho*self.S_wing*self.c*self.C_m_alpha*w) / (2*self.jy) + ((self.rho*self.S_wing*(self.c**2)*self.C_m_q*q*u) / (4*self.jy*va))
        Mw = (w*self.rho*self.S_wing*self.c / self.jy)*(self.C_m_0 + self.C_m_alpha*alpha + self.C_m_delta_e*delta_e) + (self.rho*self.S_wing*self.c*self.C_m_alpha*u) / (2*self.jy) + ((self.rho*self.S_wing*(self.c**2)*self.C_m_q*q*w) / (4*self.jy*va))
        Mq = (self.rho*va*self.S_wing*(self.c**2)*self.C_m_q) / (4*self.jy)
        Mde = (self.rho*(va**2)*self.S_wing*self.c*self.C_m_delta_e) / (2*self.jy)

        # Lateral matrix
        Alat = np.array([[Yv, Yp, Yr, self.g*np.cos(theta)*np.cos(phi), 0.0],
                         [Lv, Lp, Lr, 0.0, 0.0],
                         [Nv, Np, Nr, 0.0, 0.0],
                         [0.0, 1.0, np.cos(phi)*np.tan(theta), q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta), 0.0],
                         [0.0, 0.0, np.cos(phi) / np.cos(theta), p*np.cos(phi)/np.cos(theta) - r*np.sin(phi)/np.cos(theta), 0.0]])
        Blat = np.array([[Yda, Ydr],
                         [Lda, Ldr],
                         [Nda, Ndr],
                         [0.0, 0.0],
                         [0.0, 0.0]])


        print('Alat Matrix: ', Alat)
        print('Blat Matrix: ', Blat)

        print('\n')

        evalue_lat, evect_lat = np.linalg.eig(Alat)
        print('Eigenvalues lat:  ', evalue_lat)
        print('\n')

        # Longitudinal matrix
        Alon = np.array([[Xu, Xw, Xq, -self.g*np.cos(theta), 0.0],
                         [Zu, Zw, Zq, -self.g*np.sin(theta), 0.0],
                         [Mu, Mw, Mq, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0],
                         [np.sin(theta), -np.cos(theta), 0.0, u*np.cos(theta) + w*np.sin(theta), 0.0]])
        Blon = np.array([[Xde, Xdt],
                         [Zde, 0.0],
                         [Mde, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0]])

        print('Along Matrix: ', Alon)
        print('Blong Matrix: ', Blon)

        print('\n')

        evalue_lon, evect_lon = np.linalg.eig(Alon)
        print('Eigenvalues lon:  ', evalue_lon)
        print('\n')

        return evalue_lat, evalue_lon

    def modes(self, states, delta, evalue_lat, evalue_lon):
        # Finding eigenvalues for modes
        # Longitudinal modes
        zeta_sp = np.sqrt(1/(1+(evalue_lon.item(1).imag/evalue_lon.item(1).real)**2))
        zeta_ph = np.sqrt(1/(1+(evalue_lon.item(3).imag/evalue_lon.item(3).real)**2))
        omega_sp = -(evalue_lon.item(1).real) / zeta_sp
        omega_ph = -(evalue_lon.item(3).real) / zeta_ph
        ts_sp = 4 / (zeta_sp*omega_sp)
        ts_ph = 4 / (zeta_ph * omega_ph)
        print('Short Period Mode: ', evalue_lon.item(1), evalue_lon.item(2))
        print('Phugoid Mode: ', evalue_lon.item(3), evalue_lon.item(4))

        # Lateral modes
        print('Spiral-divergence mode: ', evalue_lat.item(4))
        print('Roll mode: ', evalue_lat.item(1))
        print('Dutch-roll mode: ', evalue_lat.item(2), evalue_lat.item(3))

        return


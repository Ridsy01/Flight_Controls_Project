import sys

sys.path.append('..')
import numpy as np

######################################################################################
#  Inertial Params
######################################################################################
# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 400.0  # End time of simulation
Ts = 0.1  # sample time for simulation
t_plot = 0.5  # the plotting and animation is updated at this rate

ts_simulation = 0.01
start_time = 0.0
end_time = 400
ts_plotting = 0.5
ts_video = 0.5
ts_control = ts_simulation

# Inertial parameters
jx = 0.824
jy = 1.135
jz = 1.759
# jxz = 0.120
jxz = 0
g = 0.028
mass = 5488.4677
pn0 = 0.0
pe0 = 0.0
pd0 = 0.0
psi0 = 0.0
theta0 = 0.0
phi0 = 0.0
u0 = 0.0
v0 = 0.0
w0 = 0.0
p0 = 0.0
q0 = 0.0
r0 = 0.0

# inital states
states0 = np.array([[0],  # pn
                  [5.],  # pe
                  [5.],  # pd
                  [0],  # u
                  [0],  # v
                  [0],  # w
                  [0],  # phi
                  [0],  # theta
                  [0],  # psi
                  [0],  # p
                  [0],  # q
                  [0]])  # r

statestie0 = np.array([[-50],  # pn
                  [0],  # pe
                  [0.0],  # pd
                  [0],  # u
                  [0],  # v
                  [0],  # w
                  [0],  # phi
                  [0],  # theta
                  [0],  # psi
                  [0],  # p
                  [0],  # q
                  [0]])  # r
Va0 = 0.0
wn = 0.0
we = 0.0
wd = 0.0

F_max = 10000000

## aerodynamic parameters
S_wing = 21.9251
b = 11.286744  # 10.7
c = 0.19
S_prop = 0.2027
rho = 0.8
e = 0.9
AR = b ** 2 / S_wing
C_L_0 = 0.23
C_D_0 = 0.043
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.030
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99
M = 50
alpha0 = 0.47
epsilon = 0.16
C_D_p = 0.0
C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_ell_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_ell_p = -0.51  # ell=p
C_n_p = -0.069
C_Y_r = 0.0
C_ell_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_ell_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_ell_delta_r = 0.0024
C_n_delta_r = -0.069
C_prop = 1
k_motor = 80  # 80
k_tp = 0.0
k_omega = 0.0

gamma = jx * jz - jxz ** 2
gamma1 = (jxz * (jx - jy + jz)) / gamma
gamma2 = (jz * (jz - jy) + jxz ** 2) / gamma
gamma3 = jz / gamma
gamma4 = jxz / gamma
gamma5 = (jz - jx) / jy
gamma6 = jxz / jy
gamma7 = ((jx - jy) * jx + jxz ** 2) / gamma
gamma8 = jx / gamma

# Cs
cp0 = gamma3 * C_ell_0 + gamma4 * C_n_0
cpb = gamma3 * C_ell_beta + gamma4 * C_n_beta
cpp = gamma3 * C_ell_p + gamma4 * C_n_p
cpr = gamma3 * C_ell_r + gamma4 * C_n_r
cpda = gamma3 * C_ell_delta_a + gamma4 * C_n_delta_a
cpdr = gamma3 * C_ell_delta_r + gamma4 * C_n_delta_r
cr0 = gamma4 * C_ell_0 + gamma8 * C_n_0
crb = gamma4 * C_ell_beta + gamma8 * C_n_beta
crp = gamma4 * C_ell_p + gamma8 * C_n_p
crr = gamma4 * C_ell_r + gamma8 * C_n_r
crda = gamma4 * C_ell_delta_a + gamma8 * C_n_delta_a
crdr = gamma4 * C_ell_delta_r + gamma8 * C_n_delta_r


import numpy as np
import Assignment1.Plane.aerosonde_parameters as P
from Assignment1.Plane.planeanimation import plane_animation
import Assignment1.parameters.simulation_parameters as SIM
from Assignment1.tools.signalGenerator import signalGenerator
#from Assignment1.Plane.dataPlotter import dataPlotter
from Assignment1.Plane.dataplotter_auto import dataPlotter
from Assignment1.Plane.deltSliders import sliders
from Assignment1.Plane.planeDynamics import planeDynamics
from Assignment1.Plane.forcesmoments import forcemoment
from Assignment1.Plane.compute_trim import ComputeTrim
from Assignment1.Plane.compute_gains import computegains
from Assignment1.Plane.autopilot import autopilot

# Position  and attitude of the 3D object
plane_anim = plane_animation()
# my_slider = sliders()
data_plot = dataPlotter()
plane_d = planeDynamics(SIM.ts_simulation, P.states0)
plane_fm = forcemoment()
plane_trim = ComputeTrim()
plane_tf = computegains()
ap = autopilot()

temp = signalGenerator(amplitude=0.5, frequency=0.1)
Va_command = signalGenerator(y_offset=25.0, amplitude=3.0, start_time=2.0, frequency=0.01)
h_command = signalGenerator(y_offset=100.0, amplitude=10.0, start_time=0.0, frequency=0.02)
chi_command = signalGenerator(y_offset=np.radians(0.0), amplitude=np.radians(45.0), start_time=8.0, frequency=0.01)

# trim conditions
Va0 = 35.
Y = 0.0  #np.deg2rad(10)
R = np.inf

x_trim, delta = plane_trim.compute_trim(Va0, Y, R)
k_values = plane_tf.transfer_fun(x_trim, delta)

pn = 0.0
pe = 0.0
pd = 00.0
u = x_trim.item(3)
v = x_trim.item(4)
w = x_trim.item(5)
phi = x_trim.item(6)
theta = x_trim.item(7)
psi = x_trim.item(8)
p = x_trim.item(9)
q = x_trim.item(10)
r = x_trim.item(11)

# Short Period:
#u = u +10
#q = q+10
# Phugoid:
#theta = theta*10
#u=u-10
#q = q+10
# Rolling:
#p = p+10
# Spiral:
#r = r+40
# Dutch:
# phi = phi+10
# r = r*10
# p = p*10
#
states = np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r])
state0 = np.array([[pn], [pe], [pd], [u], [v], [w], [phi], [theta], [psi], [p], [q], [r]])
plane_d.state = np.ndarray.copy(state0)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")

while sim_time < SIM.end_time:
    # autopilot
    t = sim_time
    u = plane_d.state.item(3)
    v = plane_d.state.item(4)
    w = plane_d.state.item(5)
    phi = plane_d.state.item(6)
    theta = plane_d.state.item(7)
    chi = plane_d.state.item(8)
    p = plane_d.state.item(9)
    q = plane_d.state.item(10)
    r = plane_d.state.item(11)
    h = plane_d.state.item(2)
    if sim_time > 16.0:
        Va_c = 50.0  #chi_command.square(sim_time)
    else:
        Va_c = 35.0
    if sim_time > 16.0:
        h_c = 70.0  #chi_command.square(sim_time)
    else:
        h_c = h_command.square(sim_time)
    if sim_time > 6.0 and sim_time < 19.0:
        chi_c = np.deg2rad(45)  #chi_command.square(sim_time)
    elif sim_time > 19.0:
        chi_c = np.deg2rad(15)
    else:
        chi_c = 0.0
    u = np.array([[t], [u], [v], [w], [phi], [theta], [chi], [p], [q], [r], [h], [Va_c], [h_c], [chi_c]])

    delta, command_state = ap.autopilot(u, k_values)
    r = np.array([[command_state.item(0)], [command_state.item(1)], [command_state.item(2)], [h_c], [Va_c]])

    # forces and states
    fm = plane_fm.f_m(plane_d.state, delta)
    y = plane_d.update(fm)
    plane_anim.update(plane_d.state)
    Va = np.sqrt(plane_d.state.item(3)**2 + plane_d.state.item(4)**2 + plane_d.state.item(5)**2)
    data_plot.update(sim_time, plane_d.state, r, Va)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

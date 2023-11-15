from Project.viewers.X_animation import Xanimation
# blocks
from Project.viewers.X_animation import oneblock
import Project.parameters.X_parameters as P
from Project.tools.signalGenerator import signalGenerator
from Project.viewers.dataPlotter import dataPlotter
from Project.tools.ForceMomentSliders import sliders
from Project.dynamics.X_dynamics import XDynamics
from Project.dynamics.TIE_dynamics import TIEDynamics

from Project.dynamics.Tie_Fighter_PD import PDctrltie
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from matplotlib import patches as shape
from Project.dynamics import TIE_controller

# Position  and attitude of the 3D object
X_anim = Xanimation()
PD = PDctrltie()

# my_slider = sliders()
#data_plot = dataPlotter()
X_d = XDynamics(P.ts_simulation, P.states0)
Tie_d = TIEDynamics(P.ts_simulation, P.statestie0)
temp = signalGenerator(amplitude=0.5, frequency=0.1)

# initialize the simulation time
sim_time = P.start_time
fx = 0.0
fy = 0.0
fz = 0.0
l = 0.0
m = 0.0
n = 0.0
failure = 0.0
# main simulation loop
print("Press Command-Q to exit...")

x_vel = np.array([[150], [300], [450], [600], [750]])
x_vel_level = 0

p_shadow = shape.Rectangle([0,0], 5, 5, angle=-90.0, rotation_point='xy', facecolor='gray')

def satFun(angle):
    if angle > np.deg2rad(45):
        angle = np.deg2rad(45)
    elif angle < np.deg2rad(-45):
        angle = np.deg2rad(-45.)
    return angle


while sim_time < P.end_time:
    six = oneblock()

    if keyboard.is_pressed("ctrl"):
        x_vel_level += -1
        if x_vel_level <= 0:
            x_vel_level = 0
        X_d.state[3][0] = x_vel[x_vel_level]
    elif keyboard.is_pressed("shift"):
        x_vel_level += 1
        if x_vel_level >= 4:
            x_vel_level = 4
        X_d.state[3][0] = x_vel[x_vel_level]
    elif keyboard.is_pressed("right arrow"):
        X_d.state[1][0] += 1
        X_d.state[6][0] += np.deg2rad(5)
        X_d.state[6][0] = satFun(X_d.state[6][0])
    elif keyboard.is_pressed("left arrow"):
        X_d.state[1][0] -= 1
        X_d.state[6][0] -= np.deg2rad(5)
        X_d.state[6][0] = satFun(X_d.state[6][0])
    elif keyboard.is_pressed("down arrow"):
        X_d.state[7][0] += np.deg2rad(-5)
        X_d.state[2][0] += 1
        X_d.state[7][0] = satFun(X_d.state[7][0])
    elif keyboard.is_pressed("up arrow"):
        X_d.state[7][0] -= np.deg2rad(-5)
        X_d.state[2][0] -= 1
        X_d.state[7][0] = satFun(X_d.state[7][0])
    elif keyboard.is_pressed("space"):
        X_d.state[6][0] = 0
        X_d.state[7][0] = 0
        fz = 0
    elif keyboard.is_pressed("esc"):
        break
    else:
        X_d.state[6][0] = 0
        X_d.state[7][0] = 0

    X_d.state[3][0] = x_vel[x_vel_level]
    Tie_d.state[3][0]= 45.

    fz1, fy1 = PD.update(Tie_d.state[2][0],X_d.state[2][0],Tie_d.state[1][0],X_d.state[1][0],Tie_d.state[4][0],Tie_d.state[5][0])

    f_wing = np.array([[fx], [fy], [fz], [l], [m], [n]])

    f_m = np.array([[fx], [fy1], [fz1], [l], [m], [n]])

    y = X_d.update(f_wing)
    y1 = Tie_d.update(f_m)

    p_shadow = X_anim.update(X_d.state,Tie_d.state, failure, p_shadow)
    #data_plot.update(sim_time, Tie_d.state, X_d.state)

    nose_down = X_d.state[2][0]-(8*np.sin(X_d.state[7][0]))
    nose_up = X_d.state[2][0]-(8*np.sin(X_d.state[7][0]))
    wing_left = X_d.state[1][0]-(5*np.cos(X_d.state[6][0]))
    wing_right = X_d.state[1][0]+(5*np.cos(X_d.state[6][0]))

    if (nose_down >= 20. or nose_up <= -20. or wing_left <= -20. or wing_right >= 20.) and sim_time > 0.0:
        for k in range(100):
            failure = 1.0
            X_anim.update(X_d.state,Tie_d.state, failure, p_shadow)

            plt.pause(20)

    # -------increment time-------------
    sim_time += P.ts_simulation

import sys
import time
sys.path.append('.')  # one directory up
# necessary import statements
import numpy as np
import keyboard
import matplotlib.pyplot as plt
import parameters.simulation_parameters as SIM
from viewers.flight_animation import flight_animation
from tools.signalGenerator import signalGenerator
from viewers.Data_Plotter_Flight_Mech import dataPlotter
from dynamics.Flight_Dynamics import FlightDynamics
from dynamics.Flight_Dynamics import Forcesmoments

# array that makes the z direction proper
state = np.array([[0], [0], [-1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
flight_anim = flight_animation(state, scale=5)
data_plot = dataPlotter()
temp = signalGenerator(amplitude=0.5, frequency=0.1)
Dynamics = FlightDynamics()
Forces = Forcesmoments()

# initialize the simulation time
sim_time = SIM.start_time
delta_e = 0
delta_a = 0
delta_t = 0
delta_r = 0

# main simulation loop
print("Press Command-Q to exit...")

Va = 0.0
aoa = 0.0
Beta = 0.0
# while loop running the simulation
while sim_time < SIM.end_time:

    if keyboard.is_pressed("down arrow"): fx -= 1000
    if keyboard.is_pressed("up arrow"): fx += 1000
    if keyboard.is_pressed("right arrow"): fy += 1000
    if keyboard.is_pressed("left arrow"): fy -= 1000
    if keyboard.is_pressed("z"): fz += 1000
    if keyboard.is_pressed("x"): fz -= 1000
    if keyboard.is_pressed("space"): fx = 0; fy = 0; fz = 0

    elif keyboard.is_pressed("up arrow") and counter == 2:
        X_d.state[3][0] = 50
        counter = counter + 1
    elif keyboard.is_pressed("up arrow") and counter == 3:
        X_d.state[3][0] = 75
        counter = counter + 1

    if keyboard.is_pressed("down arrow"): delta_e -= np.deg2rad(.5)
    if keyboard.is_pressed("up arrow"): delta_e += np.deg2rad(.5)
    if keyboard.is_pressed("right arrow"): delta_a += np.deg2rad(0.1)
    if keyboard.is_pressed("left arrow"): delta_a -= np.deg2rad(0.1)
    if keyboard.is_pressed("a"): delta_r += np.deg2rad(0.1)
    if keyboard.is_pressed("d"
                           ""): delta_r -= np.deg2rad(0.1)
    if keyboard.is_pressed("shift"): delta_t += 0.05
    if keyboard.is_pressed("left control"): delta_t -= 0.05
    if keyboard.is_pressed("space"): delta_e = 0; delta_a = 0; delta_r = 0

    if delta_t >= 1:
        delta_t = 1
    elif delta_t <= 0:
        delta_t = 0

    force, moment, Va, aoa, Beta = Forces.forces(Dynamics.state, delta_e, delta_a, delta_r, delta_t, Va)
    y = Dynamics.update(force[0], force[1], force[2], moment[0], moment[1], moment[2])
    flight_anim.update(Dynamics.state)
    data_plot.update(sim_time, Dynamics.state, Va, delta_e, delta_a, delta_r, delta_t)
    sim_time += SIM.ts_simulation
"""
Class for plotting a uav

Author: Raj #
"""
import sys

sys.path.append('')  # one directory up
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class sliders():
    def __init__(self):
        self.flag_init = True
        self.fig = plt.figure(1)
        self.axdelt_e = plt.axes([0.25, 0.055, 0.65, 0.03])
        self.delt_e = 0
        self.delt_e_slider = Slider(
            ax=self.axdelt_e,
            label='Elevator',
            valmin=-0.5,
            valmax=0.5,
            valstep=0.05,
            valinit=0.0,
        )
        self.delt_e_slider.on_changed(self.update)

        self.axdelt_a = plt.axes([0.25, 0.03, 0.65, 0.03])
        self.delt_a_slider = Slider(
            ax=self.axdelt_a,
            label="Aileron",
            valmin=-0.1,
            valmax=0.1,
            valinit=0,
            valstep=0.01,
        )
        self.delt_a_slider.on_changed(self.update)

        self.axdelt_r = plt.axes([0.25, 0.005, 0.65, 0.03])  # location of the slider on figure
        self.delt_r_slider = Slider(
            ax=self.axdelt_r,
            label="Rudder",
            valmin=-0.1,
            valmax=0.1,
            valinit=0.0,
            valstep=0.01,
        )
        self.delt_r_slider.on_changed(self.update)

        self.ax_delt_t = plt.axes([0.05, 0.25, 0.0225, 0.63])
        self.delt_t_slider = Slider(
            ax=self.ax_delt_t,
            label='Thrust',
            valmin=-1,
            valmax=1,
            valstep=0.1,
            valinit=0.0,
            orientation="vertical"
        )
        self.delt_t_slider.on_changed(self.update)


    def update(self, val):
        self.delt_t = self.delt_t_slider.val
        self.delt_e = self.delt_e_slider.val
        self.delt_a = self.delt_a_slider.val
        self.delt_r = self.delt_r_slider.val

        # plt.pause(0.001)
        self.fig.canvas.draw_idle()

        # self.update(state0)

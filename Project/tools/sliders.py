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
        self.axroll = plt.axes([0.25, 0.055, 0.65, 0.03])
        self.roll = 0
        self.roll_slider = Slider(
            ax=self.axroll,
            label='Roll',
            valmin=-np.pi,
            valmax=np.pi,
            valstep=0.1,
            valinit=0,
        )
        self.roll_slider.on_changed(self.update)

        self.axyaw = plt.axes([0.25, 0.03, 0.65, 0.03])
        self.yaw_slider = Slider(
            ax=self.axyaw,
            label="Yaw",
            valmin=-np.pi,
            valmax=np.pi,
            valinit=0,
            valstep=0.01,
        )
        self.yaw_slider.on_changed(self.update)

        self.axptich = plt.axes([0.25, 0.005, 0.65, 0.03])  # location of the slider on figure
        self.pitch_slider = Slider(
            ax=self.axptich,
            label="Pitch",
            valmin=-np.pi / 2,
            valmax=np.pi / 2,
            valinit=0,
            valstep=0.1,
        )
        self.pitch_slider.on_changed(self.update)

        self.ax_north = plt.axes([0.05, 0.25, 0.0225, 0.63])
        self.north_slider = Slider(
            ax=self.ax_north,
            label='North',
            valmin=-5,
            valmax=5,
            valstep=0.1,
            valinit=0,
            orientation="vertical"
        )
        self.north_slider.on_changed(self.update)

        self.ax_east = plt.axes([0.12, 0.25, 0.0225, 0.63])
        self.east_slider = Slider(
            ax=self.ax_east,
            label='East',
            valmin=-5,
            valmax=5,
            valstep=0.1,
            valinit=0,
            orientation="vertical"
        )
        self.east_slider.on_changed(self.update)

        self.ax_down = plt.axes([0.19, 0.25, 0.0225, 0.63])
        self.down_slider = Slider(
            ax=self.ax_down,
            label="Down",
            valmin=-5,
            valmax=5,
            valinit=0,
            valstep=0.1,
            orientation="vertical"
        )
        self.down_slider.on_changed(self.update)

    def update(self, val):
        self.roll = self.roll_slider.val
        self.yaw = self.yaw_slider.val
        self.pitch = self.pitch_slider.val
        self.north = self.north_slider.val
        self.east = self.east_slider.val
        self.down = self.down_slider.val

        # plt.pause(0.001)
        self.fig.canvas.draw_idle()

        # self.update(state0)

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
        self.axl = plt.axes([0.25, 0.055, 0.65, 0.03])
        self.l = 0
        self.l_slider = Slider(
            ax=self.axl,
            label='l',
            valmin=-100,
            valmax=100,
            valstep=10,
            valinit=0.0,
        )
        self.l_slider.on_changed(self.update)

        self.axm = plt.axes([0.25, 0.03, 0.65, 0.03])
        self.m_slider = Slider(
            ax=self.axm,
            label="m",
            valmin=-100,
            valmax=100,
            valinit=0.0,
            valstep=10,
        )
        self.m_slider.on_changed(self.update)

        self.axn = plt.axes([0.25, 0.005, 0.65, 0.03])  # location of the slider on figure
        self.n_slider = Slider(
            ax=self.axn,
            label="n",
            valmin=-100,
            valmax= 100,
            valinit=0.0,
            valstep=10,
        )
        self.n_slider.on_changed(self.update)

        self.ax_fx = plt.axes([0.05, 0.25, 0.0225, 0.63])
        self.fx_slider = Slider(
            ax=self.ax_fx,
            label='Fx',
            valmin=-10000000,
            valmax=10000000,
            valstep=1000,
            valinit=0.0,
            orientation="vertical"
        )
        self.fx_slider.on_changed(self.update)

        self.ax_fy = plt.axes([0.12, 0.25, 0.0225, 0.63])
        self.fy_slider = Slider(
            ax=self.ax_fy,
            label='Fy',
            valmin=-10000000,
            valmax=10000000,
            valstep=1000,
            valinit=0.0,
            orientation="vertical"
        )
        self.fy_slider.on_changed(self.update)

        self.ax_fz = plt.axes([0.19, 0.25, 0.0225, 0.63])
        self.fz_slider = Slider(
            ax=self.ax_fz,
            label="Fz",
            valmin=-10000000,
            valmax=10000000,
            valinit=0.0,
            valstep=1000,
            orientation="vertical"
        )
        self.fz_slider.on_changed(self.update)

    def update(self, val):
        self.fx = self.fx_slider.val
        self.fy = self.fy_slider.val
        self.fz = self.fz_slider.val
        self.l = self.l_slider.val
        self.m = self.m_slider.val
        self.n = self.n_slider.val

        # plt.pause(0.001)
        self.fig.canvas.draw_idle()

        # self.update(state0)

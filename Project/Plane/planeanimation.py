import sys
sys.path.append('')  # one directory up
from math import cos, sin
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from Assignment1.tools.rotations import Euler2Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class plane_animation():
    def __init__(self, scale=0.25):

        self.scale = scale
        self.flag_init = True
        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_title('3D Animation')
        self.ax.set_xlabel('East(m)')
        self.ax.set_ylabel('North(m)')
        self.ax.set_zlabel('Height(m)')

        north_cor_x = [0, 0]
        north_cor_y = [0, 1]
        north_cor_z = [0, 0]

        east_cor_x = [0, 1]
        east_cor_y = [0, 0]
        east_cor_z = [0, 0]

        down_cor_x = [0, 0]
        down_cor_y = [0, 0]
        down_cor_z = [0, -1]
        # plot coordinate at center
        self.ax.plot(north_cor_x, north_cor_y, north_cor_z, 'r', lw=2)
        self.ax.plot(east_cor_x, east_cor_y, east_cor_z, 'b', lw=2)
        self.ax.plot(down_cor_x, down_cor_y, down_cor_z, 'g', lw=2)
        self.ax.text(0, 1, 0, 'N')
        self.ax.text(1, 0, 0, 'E')
        self.ax.text(0, 0, -1, 'D')

        # self.update(state0)

    def cube_vertices(self, pn, pe, pd, phi, theta, psi):
        w = self.scale
        V = np.array([[2.8, 0.0, 0.0],
                      [0.1, 0.5, -0.5],
                      [0.1, -0.5, -0.5],
                      [0.1, -0.5, 0.5],
                      [0.1, 0.5, 0.5],
                      [-5.0, 0.0, 0.0],
                      [0.0, 3.5, 0.0],
                      [-1.5, 3.5, 0.0],
                      [-1.5, -3.5, 0.0],
                      [0.0, -3.5, 0.0],
                      [-4.25, 1.65, 0],
                      [-5.0, 1.65, 0],
                      [-5.0, -1.65, 0.0],
                      [-4.25, -1.65, 0.0],
                      [-4.25, 0.0, 0.0],
                      [-5.0, 0.0, -1.5]])
        pos_ned = np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep = np.tile(pos_ned, (16, 1))  # 8 vertices # 21 vertices for UAV

        R = Euler2Rotation(phi, theta, psi)

        # rotate
        vr = np.matmul(R, V.T).T
        # translate
        vr = vr + ned_rep
        # rotate for plotting north=y east=x h=-z
        R_plot = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, -1]])

        vr = np.matmul(R_plot, vr.T).T

        Vl = vr.tolist()
        Vl1 = Vl[0]
        Vl2 = Vl[1]
        Vl3 = Vl[2]
        Vl4 = Vl[3]
        Vl5 = Vl[4]
        Vl6 = Vl[5]
        Vl7 = Vl[6]
        Vl8 = Vl[7]
        Vl9 = Vl[8]
        Vl10 = Vl[9]
        Vl11 = Vl[10]
        Vl12 = Vl[11]
        Vl13 = Vl[12]
        Vl14 = Vl[13]
        Vl15 = Vl[14]
        Vl16 = Vl[15]

        # f1=. #v1 v2 v3 v4
        # f2  #v5 v6 v7 v8
        # f3= #v3 v4 v8 v7
        # f4 #v2 v1 v5 v6
        # f5 #v1 v4 v8 v5
        # f6 #v3 v7 v6 v2

        verts = [[Vl1, Vl2, Vl3],  # face 1
                 [Vl1, Vl3, Vl4],  # face 2
                 [Vl1, Vl4, Vl5],  # face 3
                 [Vl1, Vl5, Vl2],  # face 4
                 [Vl2, Vl3, Vl4, Vl5],  # face 5
                 [Vl2, Vl3, Vl6],  # face 6
                 [Vl2, Vl5, Vl6],  # face 7
                 [Vl4, Vl5, Vl6],  # face 8
                 [Vl4, Vl3, Vl6],  # face 9
                 [Vl7, Vl8, Vl9, Vl10],  # face 10
                 [Vl11, Vl12, Vl13, Vl14],  # face 11
                 [Vl6, Vl15, Vl16]]  # face 12

        return verts

    def update(self, state):
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        phi = state.item(6)
        theta = state.item(7)
        psi = state.item(8)

        # draw plot elements: cart, bob, rod
        self.draw_cube(pn, pe, pd, phi, theta, psi)

        self.ax.set_xlim(-10 + pe, 10+pe)
        self.ax.set_ylim(-10 + pn, 10+pn)
        self.ax.set_zlim(-10 - pd, 10-pd)

        # Set initialization flag to False after first call
        if self.flag_init == True:
            self.flag_init = False

    def draw_cube(self, pn, pe, pd, phi, theta, psi):
        verts = self.cube_vertices(pn, pe, pd, phi, theta, psi)
        if self.flag_init is True:
            poly = Poly3DCollection(verts, facecolors=['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'r', 'r', 'k'],
                                    alpha=1)
            self.cube = self.ax.add_collection3d(poly)  #
            plt.pause(0.01)
        else:
            self.cube.set_verts(verts)
            plt.pause(0.01)

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = -self.roll
        pitch = -self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
              sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        # plt.cla() # use handle
        if self.flag_init is True:
            body, = self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0], p1_t[0], p3_t[0], p4_t[0], p2_t[0]],
                                 [p1_t[1], p2_t[1], p3_t[1], p4_t[1], p1_t[1], p3_t[1], p4_t[1], p2_t[1]],
                                 [p1_t[2], p2_t[2], p3_t[2], p4_t[2], p1_t[2], p3_t[2], p4_t[2], p2_t[2]],
                                 'k-')  # rotor
            self.handle.append(body)

            traj, = self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')  # trajectory
            self.handle.append(traj)

            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            self.ax.set_zlim(0, 4)
            plt.xlabel('North')
            plt.ylabel('East')
            self.flag_init = False
            plt.pause(0.01)  # can be put in the main file
        else:
            self.handle[0].set_data([p1_t[0], p2_t[0], p3_t[0], p4_t[0], p1_t[0], p3_t[0], p4_t[0], p2_t[0]],
                                    [p1_t[1], p2_t[1], p3_t[1], p4_t[1], p1_t[1], p3_t[1], p4_t[1], p2_t[1]])
            self.handle[0].set_3d_properties([p1_t[2], p2_t[2], p3_t[2], p4_t[2], p1_t[2], p3_t[2], p4_t[2], p2_t[2]])

            self.handle[1].set_data(self.x_data, self.y_data)
            self.handle[1].set_3d_properties(self.z_data)
            print(self.handle)
            plt.pause(0.01)

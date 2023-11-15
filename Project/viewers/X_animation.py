import sys

import matplotlib

sys.path.append('')  # one directory up
from math import cos, sin
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from tools.rotations import Euler2Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import keyboard
from matplotlib import patches as shape
import mpl_toolkits.mplot3d.art3d as art3d


class Xanimation():
    def __init__(self):

        self.flag_init = True
        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-15, 15])
        self.ax.set_ylim([-15, 15])
        self.ax.set_zlim([-15, 15])
        self.ax.set_title('3D Animation')
        self.ax.set_xlabel('East(m)')
        self.ax.set_ylabel('North(m)')
        self.ax.set_zlabel('Height(m)')
        plt.axis('off')

        x1 = [-20, -20]

        p = shape.Rectangle(x1, 10000000, 40, angle=0.0, rotation_point='xy', facecolor='red', fill=False)
        self.ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=-20, zdir="x")

        # p = shape.Rectangle(x1, 10000000, 40, angle=0.0, rotation_point='xy', facecolor='yellow', fill=False)
        # self.ax.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=20, zdir="x")

        p = shape.Rectangle(x1, 1000000, 40, angle=-90.0, rotation_point='xy', facecolor='blue', fill=False)
        self.ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=-20, zdir="z")

        self.ax.azim = -85
        self.ax.elev = 5
        self.ax.grid(False)

        p = shape.Rectangle([-20, -20], 10, 40, angle=0.0, rotation_point='xy', facecolor='red')
        self.ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=50, zdir="y")

        p = shape.Rectangle([50, -20], 10, 40, angle=0.0, rotation_point='xy', facecolor='yellow')
        self.ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=-10, zdir="x")
        k = 0.0
        for i in range(50):
            self.ax.plot([-20, 20], [k, k], [-20, -20], color='dimgrey', linewidth=2)
            k += 25.

        # self.update(state0)
    def xwing_vertices(self, pn, pe, pd, phi, theta, psi):
        V = np.array(
            [[1, -1, -1],  # base cube
             [-1, -1, -1],
             [-1, 1, -1],
             [1, 1, -1],
             [1, -1, 1],
             [-1, -1, 1],
             [-1, 1, 1],
             [1, 1, 1],
             [1, 7 / 8, 7 / 8],  # front of ship
             [1, -7 / 8, 7 / 8],
             [1, 7 / 8, -7 / 8],
             [1, -7 / 8, -7 / 8],
             [6, 1 / 2, 1 / 2],
             [6, -1 / 2, 1 / 2],
             [6, 1 / 2, -1 / 2],
             [6, -1 / 2, -1 / 2],
             [0, 1, 1],  # engine 1
             [0, 1, 3 / 2],
             [0, 3 / 2, 3 / 2],
             [0, 3 / 2, 1],
             [-2, 1, 1],
             [-2, 1, 3 / 2],
             [-2, 3 / 2, 3 / 2],
             [-2, 3 / 2, 1],
             [0, -1, 1],  # engine 2
             [0, -1, 3 / 2],
             [0, -3 / 2, 3 / 2],
             [0, -3 / 2, 1],
             [-2, -1, 1],
             [-2, -1, 3 / 2],
             [-2, -3 / 2, 3 / 2],
             [-2, -3 / 2, 1],
             [0, 1, -1],  # engine 3
             [0, 1, -3 / 2],
             [0, 3 / 2, -3 / 2],
             [0, 3 / 2, -1],
             [-2, 1, -1],
             [-2, 1, -3 / 2],
             [-2, 3 / 2, -3 / 2],
             [-2, 3 / 2, -1],
             [0, -1, -1],  # engine 4
             [0, -1, -3 / 2],
             [0, -3 / 2, -3 / 2],
             [0, -3 / 2, -1],
             [-2, -1, -1],
             [-2, -1, -3 / 2],
             [-2, -3 / 2, -3 / 2],
             [-2, -3 / 2, -1],
             [1, 5, 3],  # wing 1
             [-1, 5, 3],
             [1, -5, -3],
             [-1, -5, -3],
             [1, -5, 3],  # wing 2
             [-1, -5, 3],
             [1, 5, -3],
             [-1, 5, -3],
             [6, 5 / 8, 5 / 8],  # front triangle
             [6, -5 / 8, 5 / 8],
             [6, 5 / 8, -5 / 8],
             [6, -5 / 8, -5 / 8],
             [8, -1 / 4, 0],
             [8, 1 / 4, 0],  # end of front triangle
             [-1, 4.875, 3.125],  # start of gun 1 (bottom right)
             [-1, 4.875, 2.875],
             [-1, 5.125, 2.875],
             [-1, 5.125, 3.125],
             [4, 4.875, 3.125],
             [4, 4.875, 2.875],
             [4, 5.125, 2.875],
             [4, 5.125, 3.125],  # end of gun 1
             [-1, -4.875, 3.125],  # start of gun 2 (bottom left)
             [-1, -4.875, 2.875],
             [-1, -5.125, 2.875],
             [-1, -5.125, 3.125],
             [4, -4.875, 3.125],
             [4, -4.875, 2.875],
             [4, -5.125, 2.875],
             [4, -5.125, 3.125],  # end of gun 2
             [-1, 4.875, -3.125],  # start of gun 3 (top right)
             [-1, 4.875, -2.875],
             [-1, 5.125, -2.875],
             [-1, 5.125, -3.125],
             [4, 4.875, -3.125],
             [4, 4.875, -2.875],
             [4, 5.125, -2.875],
             [4, 5.125, -3.125],
             [-1, -4.875, -3.125],  # start of gun 4 (top left)
             [-1, -4.875, -2.875],
             [-1, -5.125, -2.875],
             [-1, -5.125, -3.125],
             [4, -4.875, -3.125],
             [4, -4.875, -2.875],
             [4, -5.125, -2.875],
             [4, -5.125, -3.125]  # end of gun 4
             ])
        pos_ned = np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep = np.tile(pos_ned, (94, 1))  # 8 vertices # 21 vertices for UAV

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
        Vl17 = Vl[16]
        Vl18 = Vl[17]
        Vl19 = Vl[18]
        Vl20 = Vl[19]
        Vl21 = Vl[20]
        Vl22 = Vl[21]
        Vl23 = Vl[22]
        Vl24 = Vl[23]
        Vl25 = Vl[24]
        Vl26 = Vl[25]
        Vl27 = Vl[26]
        Vl28 = Vl[27]
        Vl29 = Vl[28]
        Vl30 = Vl[29]
        Vl31 = Vl[30]
        Vl32 = Vl[31]
        Vl33 = Vl[32]
        Vl34 = Vl[33]
        Vl35 = Vl[34]
        Vl36 = Vl[35]
        Vl37 = Vl[36]
        Vl38 = Vl[37]
        Vl39 = Vl[38]
        Vl40 = Vl[39]
        Vl41 = Vl[40]
        Vl42 = Vl[41]
        Vl43 = Vl[42]
        Vl44 = Vl[43]
        Vl45 = Vl[44]
        Vl46 = Vl[45]
        Vl47 = Vl[46]
        Vl48 = Vl[47]
        Vl49 = Vl[48]
        Vl50 = Vl[49]
        Vl51 = Vl[50]
        Vl52 = Vl[51]
        Vl53 = Vl[52]
        Vl54 = Vl[53]
        Vl55 = Vl[54]
        Vl56 = Vl[55]
        Vl57 = Vl[56]
        Vl58 = Vl[57]
        Vl59 = Vl[58]
        Vl60 = Vl[59]
        Vl61 = Vl[60]
        Vl62 = Vl[61]
        Vl63 = Vl[62]
        Vl64 = Vl[63]
        Vl65 = Vl[64]
        Vl66 = Vl[65]
        Vl67 = Vl[66]
        Vl68 = Vl[67]
        Vl69 = Vl[68]
        Vl70 = Vl[69]
        Vl71 = Vl[70]
        Vl72 = Vl[71]
        Vl73 = Vl[72]
        Vl74 = Vl[73]
        Vl75 = Vl[74]
        Vl76 = Vl[75]
        Vl77 = Vl[76]
        Vl78 = Vl[77]
        Vl79 = Vl[78]
        Vl80 = Vl[79]
        Vl81 = Vl[80]
        Vl82 = Vl[81]
        Vl83 = Vl[82]
        Vl84 = Vl[83]
        Vl85 = Vl[84]
        Vl86 = Vl[85]
        Vl87 = Vl[86]
        Vl88 = Vl[87]
        Vl89 = Vl[88]
        Vl90 = Vl[89]
        Vl91 = Vl[90]
        Vl92 = Vl[91]
        Vl93 = Vl[92]
        Vl94 = Vl[93]

        verts = [[Vl1, Vl2, Vl3, Vl4],  # face 1
                 [Vl5, Vl6, Vl7, Vl8],
                 [Vl3, Vl4, Vl8, Vl7],
                 [Vl2, Vl1, Vl5, Vl6],
                 [Vl1, Vl4, Vl8, Vl5],
                 [Vl3, Vl7, Vl6, Vl2],  # face 6 end of base cube
                 [Vl9, Vl10, Vl11, Vl12],  # face 7 start of front of ship
                 [Vl13, Vl14, Vl15, Vl16],
                 [Vl9, Vl13, Vl14, Vl10],
                 [Vl9, Vl13, Vl15, Vl11],
                 [Vl11, Vl15, Vl16, Vl12],
                 [Vl12, Vl16, Vl14, Vl10],  # face 12 end of front of ship
                 [Vl17, Vl21, Vl24, Vl20],  # face 13 start of engine 1 (bottom right)
                 [Vl17, Vl21, Vl22, Vl18],
                 [Vl18, Vl22, Vl23, Vl19],
                 [Vl19, Vl23, Vl24, Vl20],
                 [Vl17, Vl18, Vl19, Vl20],
                 [Vl21, Vl22, Vl23, Vl24],  # face 18 end of engine 1
                 [Vl25, Vl29, Vl32, Vl28],  # face 19 start of engine 2 (bottom left)
                 [Vl25, Vl29, Vl30, Vl26],
                 [Vl26, Vl30, Vl31, Vl27],
                 [Vl27, Vl31, Vl32, Vl28],
                 [Vl25, Vl26, Vl27, Vl28],
                 [Vl29, Vl30, Vl31, Vl32],  # face 24 end of engine 2
                 [Vl33, Vl37, Vl40, Vl36],  # face 25 start of engine 3 (bottom right)
                 [Vl33, Vl37, Vl38, Vl34],
                 [Vl34, Vl38, Vl39, Vl35],
                 [Vl35, Vl39, Vl40, Vl36],
                 [Vl33, Vl34, Vl35, Vl36],
                 [Vl37, Vl38, Vl39, Vl40],  # face 30 end of engine 3 (top right)
                 [Vl41, Vl45, Vl48, Vl44],  # face 31 start of engine 4 (top left)
                 [Vl41, Vl45, Vl46, Vl42],
                 [Vl42, Vl46, Vl47, Vl43],
                 [Vl43, Vl47, Vl48, Vl44],
                 [Vl41, Vl42, Vl43, Vl44],
                 [Vl45, Vl46, Vl47, Vl48],  # face 36 end of engine 4
                 [Vl49, Vl51, Vl52, Vl50],  # face 37 wing 1
                 [Vl53, Vl55, Vl56, Vl54],  # face 38 wing 2
                 [Vl57, Vl62, Vl61, Vl58],  # face 39 start of front triangle
                 [Vl58, Vl60, Vl61],
                 [Vl57, Vl59, Vl62],
                 [Vl59, Vl62, Vl61, Vl60],  # face 42 end of front triangle
                 [Vl63, Vl64, Vl65, Vl66],  # face 43 start of gun 1
                 [Vl67, Vl68, Vl69, Vl70],
                 [Vl63, Vl67, Vl68, Vl64],
                 [Vl63, Vl67, Vl70, Vl66],
                 [Vl66, Vl70, Vl69, Vl65],
                 [Vl65, Vl69, Vl68, Vl64],  # face 48 end of gun 1
                 [Vl71, Vl72, Vl73, Vl74],  # face 49 start of gun 2
                 [Vl75, Vl76, Vl77, Vl78],
                 [Vl71, Vl75, Vl76, Vl72],
                 [Vl71, Vl75, Vl78, Vl74],
                 [Vl74, Vl78, Vl77, Vl73],
                 [Vl73, Vl77, Vl76, Vl72],  # face 54 end of gun 2
                 [Vl79, Vl80, Vl81, Vl82],  # face 55 start of gun 3
                 [Vl83, Vl84, Vl85, Vl86],
                 [Vl79, Vl83, Vl84, Vl80],
                 [Vl79, Vl83, Vl86, Vl82],
                 [Vl82, Vl86, Vl85, Vl81],
                 [Vl81, Vl85, Vl84, Vl80],  # face 60 end of gun 3
                 [Vl87, Vl88, Vl89, Vl90],  # face 61 start of gun 4
                 [Vl91, Vl92, Vl93, Vl94],
                 [Vl87, Vl91, Vl92, Vl88],
                 [Vl87, Vl91, Vl94, Vl90],
                 [Vl90, Vl94, Vl93, Vl89],
                 [Vl89, Vl93, Vl92, Vl88]]  # face 66 end of gun 4

        return (verts)

    def tie_vertices(self, pn, pe, pd, phi, theta, psi):
        V = np.array(
            [[15, -1.5, -2], # base cube (tie fighter)
            [15, 1.5, -2],
            [15, -1.5, 2],
            [15, 1.5, 2],
            [17, -1.5, -2],
            [17, 1.5, -2],
            [17, -1.5, 2],
            [17, 1.5, 2],
            [13.5, 2, -4], # Wing 1 (tie fighter)
            [13.5, 2, 4],
            [18.5, 2, -4],
            [18.5, 2, 4],
            [13.5, -2, -4], # Wing 2 (tie fighter)
            [13.5, -2, 4],
            [18.5, -2, -4],
            [18.5, -2, 4],
            [13.5, 2, -4], # Wing 3 (tie fighter)
            [18.5, 2, -4],
            [13.5, .5, -5.5],
            [18.5, .5, -5.5],
            [13.5, 2, 4], # Wing 4 (tie fighter)
            [18.5, 2, 4],
            [13.5, .5, 5.5],
            [18.5, .5, 5.5],
            [13.5, -2, -4], # Wing 5 (tie fighter)
            [18.5, -2, -4],
            [13.5, -.5, -5.5],
            [18.5, -.5, -5.5],
            [13.5, -2, 4], # Wing 6 (tie fighter)
            [18.5, -2, 4],
            [13.5, -.5, 5.5],
            [18.5, -.5, 5.5],
            [13.5, 2, 0],
            [18.5, 2, 0],
            [13.5, -2, 0],
            [18.5, -2, 0]])  # end of gun 4
        pos_ned = np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep = np.tile(pos_ned, (36, 1))  # 8 vertices # 21 vertices for UAV

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
        Vl127 = Vl[0]
        Vl128 = Vl[1]
        Vl129 = Vl[2]
        Vl130 = Vl[3]
        Vl131 = Vl[4]
        Vl132 = Vl[5]
        Vl133 = Vl[6]
        Vl134 = Vl[7]
        Vl135 = Vl[8]
        Vl136 = Vl[9]
        Vl137 = Vl[10]
        Vl138 = Vl[11]
        Vl139 = Vl[12]
        Vl140 = Vl[13]
        Vl141 = Vl[14]
        Vl142 = Vl[15]
        Vl143 = Vl[16]
        Vl144 = Vl[17]
        Vl145 = Vl[18]
        Vl146 = Vl[19]
        Vl147 = Vl[20]
        Vl148 = Vl[21]
        Vl149 = Vl[22]
        Vl150 = Vl[23]
        Vl151 = Vl[24]
        Vl152 = Vl[25]
        Vl153 = Vl[26]
        Vl154 = Vl[27]
        Vl155 = Vl[28]
        Vl156 = Vl[29]
        Vl157 = Vl[30]
        Vl158 = Vl[31]
        Vl159 = Vl[32]
        Vl160 = Vl[33]
        Vl161 = Vl[34]
        Vl162 = Vl[35]

        verts = [[Vl127,Vl128,Vl130,Vl129],  #face 91 start of tie base
                    [Vl131,Vl132,Vl134,Vl133],
                    [Vl129,Vl130,Vl134,Vl133],
                    [Vl128,Vl127,Vl131,Vl132],
                    [Vl127,Vl130,Vl134,Vl131],
                    [Vl129,Vl133,Vl132,Vl128],  # face 96 end of tie base
                    [Vl135,Vl136,Vl138,Vl137], # face 97 tie wing 1
                    [Vl139,Vl140,Vl142,Vl141], # face 98 tie wing 2
                    [Vl143,Vl144,Vl146,Vl145],
                    [Vl147,Vl148,Vl150,Vl149],
                    [Vl151,Vl152,Vl154,Vl153],
                    [Vl155,Vl156,Vl158,Vl157],
                    [Vl159,Vl160,Vl162,Vl161]]

        return (verts)



    def update(self, state,Tie_state, failure, p_shadow):
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        phi = state.item(6)
        theta = state.item(7)
        psi = state.item(8)

        # draw plot elements: cart, bob, rod
        self.draw_xwing(pn, pe, pd, phi, theta, psi)
        plt.pause(.01)
        pn1 = Tie_state.item(0)
        pe1 = Tie_state.item(1)
        pd1 = Tie_state.item(2)
        phi1 = Tie_state.item(6)
        theta1 = Tie_state.item(7)
        psi1 = Tie_state.item(8)

        self.draw_tie(pn1, pe1, pd1, phi1, theta1, psi1)
        p_shadow.set_visible(False)

        p = shape.Rectangle([pe - 2.5, pn+8], 10, 5, angle=-90.0, rotation_point='xy', facecolor='gray')
        p_shadow = shape.Shadow(p, -0.01, -.01, shade=0.01)
        self.ax.add_patch(p_shadow)
        art3d.pathpatch_2d_to_3d(p_shadow, z=-20, zdir="z")

        if failure == 1:
            for k in range(100):
                if keyboard.is_pressed("esc"):
                    plt.close()
                if (k % 2) == 0:
                    if keyboard.is_pressed("esc"):
                        plt.close()
                    self.ax.patch.set_facecolor('xkcd:tomato red')
                    plt.pause(0.7)
                else:
                    if keyboard.is_pressed("esc"):
                        plt.close()
                    self.ax.patch.set_facecolor('xkcd:pale grey')
                    plt.pause(0.7)

        # green box from kevin ##########################################

        ##################################################################

        self.ax.set_xlim(-15 + pe, 15 + pe)
        self.ax.set_ylim(-15 + pn, 15 + pn)
        self.ax.set_zlim(-15 - pd, 15 - pd)

        # Set initialization flag to False after first call
        if self.flag_init == True:
            self.flag_init = False

        return p_shadow

    def draw_xwing(self, pn, pe, pd, phi, theta, psi):
        verts = self.xwing_vertices(pn, pe, pd, phi, theta, psi)
        if self.flag_init is True:
            poly = Poly3DCollection(verts,
                                    facecolors=['b', 'lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                                'lightgray', 'lightgray', 'lightgray', 'r',
                                                'lightgray', 'r', 'gray', 'goldenrod', 'gray', 'goldenrod', 'gray', 'r',
                                                'gray', 'goldenrod',
                                                'gray', 'goldenrod', 'gray', 'r', 'gray', 'goldenrod', 'gray',
                                                'goldenrod', 'grey', 'r',
                                                'gray', 'goldenrod', 'gray', 'goldenrod', 'gray', 'r', 'lightgray',
                                                'lightgray', 'k', 'goldenrod',
                                                'goldenrod', 'k', 'k', 'r', 'gray', 'gray', 'gray', 'gray', 'k', 'r',
                                                'gray', 'gray', 'gray', 'gray', 'k', 'r', 'gray', 'gray', 'gray',
                                                'gray',
                                                'k', 'r', 'gray', 'gray', 'gray', 'gray'], alpha=1)
            self.xwing = self.ax.add_collection3d(poly)  #
            plt.pause(0.01)
        else:
            self.xwing.set_verts(verts)
            plt.pause(0.01)

    def draw_tie(self, pn, pe, pd, phi, theta, psi):
        verts = self.tie_vertices(pn, pe, pd, phi, theta, psi)
        if self.flag_init is True:
            poly = Poly3DCollection(verts,
                                    facecolors=['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'k', 'k', 'k', 'k', 'k',
                                                'k', 'k'], alpha=1)
            self.tief = self.ax.add_collection3d(poly)  #
            plt.pause(0.01)
        else:
            self.tief.set_verts(verts)
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


class oneblock:
    def __init__(self):
        import matplotlib.pyplot as plt

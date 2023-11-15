import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter:

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 2   # Number of subplot rows
        self.num_cols = 2    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        # self.phi_c_history = []  # phi c angle
        # self.theta_c_history = []  # theta c angle
        # self.theta_history = []  # theta angle
        # self.phi_history = []  # phi angle
        # self.psi_c_history = []  # psi c angle
        # self.psi_history = []  # psi angle
        self.pn_history = []  # north direction
        self.pn_c_history = []  # north c direction
        self.pe_history = []  # east direction
        self.pe_c_history = []  # east c direction
        self.pd_history = []  # down direction
        self.pd_c_history = []  # down c direction
        self.u_history = []
        # self.v_history = []
        # self.w_history = []
        # self.p_history = []
        # self.q_history = []
        # self.r_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0][0], ylabel='Pn(m)', title='X-Wing Parameters'))
        self.handle.append(myPlot(self.ax[1][0], ylabel='Pe(m)'))
        self.handle.append(myPlot(self.ax[0][1], ylabel='Pd(m)'))
        self.handle.append(myPlot(self.ax[1][1], xlabel='t(s)', ylabel='Va(m/s)'))
        # self.handle.append(myPlot(self.ax[3][0], ylabel='Phi(rad)'))
        # self.handle.append(myPlot(self.ax[4][0], ylabel='Theta(rad)'))
        # self.handle.append(myPlot(self.ax[5][0], xlabel='t(s)', ylabel='Psi(rad)'))
        # self.handle.append(myPlot(self.ax[0][1], ylabel='u(m/s)'))
        # self.handle.append(myPlot(self.ax[1][1], ylabel='v(m/s)'))
        # self.handle.append(myPlot(self.ax[2][1], ylabel='w(m/s)'))
        # self.handle.append(myPlot(self.ax[3][1], ylabel='p(rad/s)'))
        # self.handle.append(myPlot(self.ax[4][1], ylabel='q(rad/s)'))
        # self.handle.append(myPlot(self.ax[5][1], xlabel='t(s)', ylabel='r(rad/s)'))

    def update(self, t, state, r):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        # v = state.item(4)
        # w = state.item(5)
        # phi = state.item(6)
        # theta = state.item(7)
        # psi = state.item(8)
        # p = state.item(9)
        # q = state.item(10)
        # r = state.item(11)
        pn_c = r.item(0)
        pe_c = r.item(1)
        pd_c = r.item(2)


        self.time_history.append(t)  # time
        # self.theta_history.append(theta)  # theta angle
        # self.phi_history.append(phi)  # phi angle
        # self.psi_history.append(psi)  # psi angle
        self.pn_history.append(pn)  # north position
        self.pn_c_history.append(pn_c)
        self.pe_history.append(pe)  # east position
        self.pe_c_history.append(pe_c)
        self.pd_history.append(pd)  # down position
        self.pd_c_history.append(pd_c)
        self.u_history.append(u)
        # self.v_history.append(v)
        # self.w_history.append(w)
        # self.p_history.append(p)
        # self.q_history.append(q)
        # self.r_history.append(r)

        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.pn_history, self.pn_c_history])
        self.handle[1].update(self.time_history, [self.pe_history, self.pe_c_history])
        self.handle[2].update(self.time_history, [self.pd_history, self.pd_c_history])
        self.handle[3].update(self.time_history, [self.u_history])
        # self.handle[3].update(self.time_history, [self.phi_history])
        # self.handle[4].update(self.time_history, [self.theta_history])
        # self.handle[5].update(self.time_history, [self.psi_history])
        # self.handle[6].update(self.time_history, [self.u_history])
        # self.handle[7].update(self.time_history, [self.v_history])
        # self.handle[8].update(self.time_history, [self.w_history])
        # self.handle[9].update(self.time_history, [self.p_history])
        # self.handle[10].update(self.time_history, [self.q_history])
        # self.handle[11].update(self.time_history, [self.r_history])


class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
           


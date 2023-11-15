import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter:

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 5  # Number of subplot rows
        self.num_cols = 1  # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.phi_c_history = []  # theta c angle
        self.phi_history = []  # theta angle
        self.theta_history = []
        self.theta_c_history = []
        self.psi_history = []
        self.psi_c_history = []
        self.h_history = []
        self.h_c_history = []
        self.Va_history = []
        self.Va_c_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='phi(rad)', title='Angles'))
        self.handle.append(myPlot(self.ax[1], ylabel='theta(rad)'))
        self.handle.append(myPlot(self.ax[2], ylabel='psi(rad)'))
        self.handle.append(myPlot(self.ax[3], ylabel='Va(m/s)'))
        self.handle.append((myPlot(self.ax[4], xlabel='t(s)',ylabel='h(m)')))

    def update(self, t, state, ref, Va):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        pn = state.item(0)
        pe = state.item(1)
        pd = -state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        phi = state.item(6)
        theta = state.item(7)
        psi = state.item(8)
        p = state.item(9)
        q = state.item(10)
        r = state.item(11)

        phi_c = ref.item(0)
        theta_c = ref.item(1)
        psi_c = ref.item(2)
        h_c = ref.item(3)
        Va_c = ref.item(4)


        self.time_history.append(t)  # time
        self.phi_history.append(phi)  # theta angle
        self.phi_c_history.append(phi_c)
        self.theta_history.append(theta)
        self.theta_c_history.append(theta_c)
        self.psi_history.append(psi)
        self.psi_c_history.append(psi_c)
        self.h_history.append(pd)
        self.h_c_history.append(h_c)
        self.Va_history.append(Va)
        self.Va_c_history.append(Va_c)


        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.phi_history, self.phi_c_history])
        self.handle[1].update(self.time_history, [self.theta_history, self.theta_c_history])
        self.handle[2].update(self.time_history, [self.psi_history, self.psi_c_history])
        self.handle[3].update(self.time_history, [self.Va_history, self.Va_c_history])
        self.handle[4].update(self.time_history, [self.h_history, self.h_c_history])


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
        self.ax = ax  # Axes handle
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
        else:  # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
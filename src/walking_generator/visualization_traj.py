import os
import sys
import re
import numpy
import json
from time import strftime
from copy import deepcopy

import matplotlib
from matplotlib import pyplot as plt

class PlotData(object):
    """
    Smart data container for saving plotting relevant data.
    """

    def __init__(self, generator):
        """ build data structures """
        self.data = {}
        self.generator = generator

        # get list keys
        self.hull_keys = generator._hull_keys
        self.plot_keys = generator._plot_keys
        self.data_keys = generator._data_keys

        # fill internal data with keys and empty lists
        for key in self.hull_keys:
            self.data[key] = []

        for key in self.data_keys:
            self.data[key] = []

        for key in self.plot_keys:
            self.data[key] = []

    def update(self):
        """ update internal data from generator """
        for key in self.data:
            val = self.generator.__dict__.get(key, [])
            self.data[key].append(deepcopy(val))

    def reset(self):
        """ reset all internal data """
        for key in self.data:
            self.data[key] = []

    def save_to_file(self, filename=''):
        """
        save data in json format to file

        Parameters
        ----------

        filename: str
            path to output file,
        """
        # generate general filename
        if not filename:
            stamp = strftime("%Y-%m-%d-%H-%M-%S")
            name = '{stamp}_generator_data.json'.format(stamp=stamp)
            filename = os.path.join('/tmp', name)

        # convert numpy arrays into lists
        for key in self.data.keys():
            val = numpy.asarray(self.data[key])
            self.data[key] = val.tolist()

        # save data to file in json format
        with open(filename, 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=2)


class PlotterTraj(object):
    """
    Real time PlotterTraj for pattern generator data. Can create plots online or
    from saved data in json format.
    """

    # counter for pictures
    picture_cnt = 0

    # polygons that should be plotted
    polygons_mapping = {
        'f_k_x' : {
            'lfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
            #'rfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
            'lfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
            #'rfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        },
        #'f_k_y' : {
        #    'lfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
        #    #'rfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
        #    'lfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        #    #'rfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        #},
        'F_k_x' : {
            'lfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
            #'rfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
            'lfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
            #'rfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        },
        #'F_k_y' : {
        #    'lfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
        #    #'rfoot'     : {'edgecolor':'gray', 'lw':1, 'fill':None,},
        #    'lfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        #    #'rfcophull' : {'edgecolor':'blue', 'lw':1, 'fill':None,},
        #},
        #'f_k_x' : ('lfposhull', 'rfposhull'),
        }

    # general mapping for the bird's eye plots
    bird_view_mapping = (
        # CoM
        (
            ('c_k_x',   {'lw':'1', 'ls':'-',  'marker':'.', 'ms':4, 'c':'r', 'label':'$c_{k}^{x}$'}),
            ('c_k_y',   {'lw':'1', 'ls':'-.', 'marker':'.', 'ms':4, 'c':'r', 'label':'$c_{k}^{y}$'}),
            # for rotation
            ('c_k_q',   {'lw':'1', 'ls':'', 'marker':'.', 'ms':4, 'c':'r', 'label':'$c_{k}^{\\theta}$'}),
        ),
        # Feet
        (
            ('f_k_x',   {'lw':'1', 'ls':'-',  'marker':'x', 'ms':4, 'c':'g', 'label':'$f_{k}^{x}$'}),
            ('f_k_y',   {'lw':'1', 'ls':'-.', 'marker':'x', 'ms':4, 'c':'g', 'label':'$f_{k}^{y}$'}),
            # for rotation
            ('f_k_q',   {'lw':'1', 'ls':'',  'marker':'x', 'ms':4, 'c':'g', 'label':'$f_{k}_{\\theta}$'}),
        ),
        # ZMP
        # TODO how to get current ZMP state?
        (
            ('z_k_x',   {'lw':'1', 'ls':'-',  'marker':'.', 'ms':4, 'c':'b', 'label':'$z_{k}^{x}$'}),
            ('z_k_y',   {'lw':'1', 'ls':'-.', 'marker':'.', 'ms':4, 'c':'b', 'label':'$z_{k}^{y}$'}),
        ),
    )

    preview_mapping = (
        # Preview
        (
            ('C_kp1_x', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{x}$'}),
            ('C_kp1_y', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{y}$'}),
            # for rotation
            ('C_kp1_q', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{\\theta}$'}),
        ),
        (
            ('F_k_x', {'lw':'1', 'ls':':', 'marker':'x', 'ms':4, 'c':'k', 'label':'$F_{k}^{x}$'}),
            ('F_k_y', {'lw':'1', 'ls':':', 'marker':'x', 'ms':4, 'c':'k', 'label':'$F_{k}^{y}$'}),
            # for rotation
            ('F_k_q', {'lw':'1', 'ls':':', 'marker':'x', 'ms':4, 'c':'k', 'label':'$F_{k}^{\\theta}$'}),
        ),
        (
            ('Z_kp1_x', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'b', 'label':'$Z_{k+1}^{x}$'}),
            ('Z_kp1_y', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'b', 'label':'$Z_{k+1}^{y}$'}),
        ),
    )

    data_mapping = (
        # Preview
        (
            ('ori_qp_nwsr', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{x}$'}),
            ('pos_qp_nwsr', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{y}$'}),
        ),
        (
            ('ori_qp_cputime}', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{x}$'}),
            ('pos_qp_cputime}', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{y}$'}),
        ),
        #(
            #('qp_nwsr', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{x}$'}),
            #('qp_nwsr', {'lw':'1', 'ls':':', 'marker':'.', 'ms':4, 'c':'r', 'label':'$C_{k+1}^{y}$'}),
        #),
    )

    def __init__(self,
        generator=None, trajectory=None, show_canvas=True, save_to_file=False, filename='',
        fmt='png', dpi=200, limits=None
    ):
        """
        Real time PlotterTraj for pattern generator data. Can create plots online or
        from saved data in json format.

        Parameters
        ----------

        generator: BaseGenerator instance
            A generator instance which enables online plotting.

        show_canvas: bool
            Flag enabling visualization using a matplotlib canvas.

        save_to_file: bool
            Flag enabling saving pictures as files.

        filename: str
            Path to output file. Note that a counter will be added to filename!

        fmt: str
            File format accepted from matplotlib. Defaults to 'png'.

        """
        # see_to_filetting matplotlib to pretty printing
        matplotlib.rc('xtick', labelsize=6)
        matplotlib.rc('ytick', labelsize=6)
        matplotlib.rc('font', size=10)
        matplotlib.rc('font', size=10)
        matplotlib.rc('text', usetex=False)

        # some plotting options
        #self.figsize = (8.0, 5.0)
        self.dpi = dpi

        # save reference to the trajectory to follow
        self.trajectory = trajectory

        # save reference to pattern generator
        # NOTE used for online plotting
        self.generator = generator

        # transformation matrix used to rotated feet hull
        self.T = numpy.zeros((2,2))

        # try to get data from generator or empty data else
        if generator:
            try:
                self.data = self.generator.data.data
            except:
                err_str = 'could not access data in generator, initializing empty data set.'
                sys.stderr.write(err_str + '\n')
                self.data = {}
        else:
            self.data = {}

        # decides if plots are shown in a canvas
        self.show_canvas = show_canvas

        # decides if plots are saved as pictures
        self.save_to_file = save_to_file

        if self.save_to_file:
            # generate general filename
            if not filename:
                stamp = strftime("%Y-%m-%d-%H-%M-%S")
                directory = '{stamp}_plots'.format(stamp=stamp)
                path = os.path.join('/tmp', directory)
                name = 'pattern_plot'
            else:
                path = os.path.dirname(filename)
                name = os.path.basename(filename).strip()
                name = name.split('.')[0]

            # extract path and filename
            self._fpath = path
            self._fname = name
            self._ffmt  = fmt

            # create path
            if not os.path.isdir(self._fpath):
                os.makedirs(self._fpath)

        # BIRD'S EYE VIEW
        # initialize figure with proper size
        self.fig = plt.figure()

        ax = self.fig.add_subplot(1,1,1)
        ax.plot(trajectory[0],trajectory[1])
        ax.set_title('Aerial View')
        ax.set_ylabel('y [m]')
        ax.set_xlabel("x [m]")

        # assemble different trajectories
        self.bird_view_axis  = ax
        self.bird_view_background = self.fig.canvas.copy_from_bbox(ax.bbox)
        self.bird_view_lines = {}
        self.bird_view_polys = {}

        self.bird_view_limits = limits

        if not self.bird_view_limits:
            self.bird_view_axis.relim()
            self.bird_view_axis.autoscale_view()
        else:
            self.bird_view_axis.set_xlim(self.bird_view_limits[0])
            self.bird_view_axis.set_ylim(self.bird_view_limits[1])

        self.bird_view_axis.set_aspect('equal')

        for item in self.bird_view_mapping:
            # get mapping for x values
            name      = item[0][0]
            settings  = item[0][1]

            # layout line with empty data, but right settings
            # remove identifier from label
            settings['label'] = re.sub(r'_{[xy]}', '', settings['label'])
            line, = ax.plot([], [], **settings)

            # store lines for later update
            self.bird_view_lines[name] = line

        # add current preview to plot
        for item in self.preview_mapping:
            # get mapping for x values
            name      = item[0][0]
            settings  = item[0][1]

            # layout line with empty data, but right settings
            # line, = ax.plot([], [], **settings)

            # store lines for later update
            self.bird_view_lines[name] = line

        # add current preview to plot
        for key in self.polygons_mapping.keys():
            self.bird_view_polys[key] = {}
            for key1 in self.polygons_mapping[key].keys():
                self.bird_view_polys[key][key1] = {}

        # show plot canvas with tght layout
        self.fig.tight_layout()
        if self.show_canvas:
            self.fig.show()

    def filename(self):
        """
        define filename as function from name, counter and format
        """
        name = self._fname
        cnt  = self.picture_cnt
        fmt  = self._ffmt
        return '{name}{cnt:04d}.{fmt}'.format(name=name, cnt=cnt, fmt=fmt)

    def _save_to_file(self, figure=None):
        """ save figure as pdf """
        # set options
        plot_options = {
            'format' : self._ffmt,
            'dpi' : self.dpi,
            'bbox_inches' : 'tight',
        }

        # convert numpy arrays into lists
        f_path = os.path.join(self._fpath, self.filename())
        if not figure:
            self.fig.savefig(f_path, **plot_options)
        else:
            figure.savefig(f_path, **plot_options)
        self.picture_cnt += 1

    def load_from_file(self, filename):
        """
        load data from file

        .. NOTE: Expects data to be in json format!

        Parameters
        ----------

        filename: str
            path to file that is opened via open(filename, 'r')

        """
        # check if file exists
        if not os.path.isfile(filename):
            err_str = 'filename is not a proper path to file:\n filename = {}'.format(filename)
            raise IOError(err_str)

        self.input_filename = filename

        # try to read data
        with open(self.input_filename, 'r') as f:
            self.data = json.load(f)

    def update(self):
        """ creates plot of x/y trajectories on the ground """

        time = numpy.asarray(self.data['time'])

        # BIRD'S EYE VIEW
        blend = {
            'c_k_y' : [],
            'c_k_x' : [],
            'f_k_y' : [],
            'f_k_x' : [],
            'z_k_y' : [],
            'z_k_x' : [],
        }

        for item in self.bird_view_mapping:
            # get names from mapping
            x_name = item[0][0]
            y_name = item[1][0]
            q_name = None
            # get theta name only if defined
            if len(item) > 2:
                q_name = item[2][0]

            # get line
            line = self.bird_view_lines[x_name]

            # define data
            x_data = numpy.ones(time.shape[0])*numpy.nan
            y_data = numpy.ones(time.shape[0])*numpy.nan

            # assemble data
            for i in range(time.shape[0]):
                # x value
                val = numpy.asarray(self.data[x_name])
                if len(val.shape) > 1:
                    val = val[i,0]
                else:
                    val = val[i]
                x_data[i] = val

                # y value
                val = numpy.asarray(self.data[y_name])
                if len(val.shape) > 1:
                    val = val[i,0]
                else:
                    val = val[i]
                y_data[i] = val

                # draw CoP and foot position hull
                for poly_name, poly_map in self.polygons_mapping.get(x_name, {}).iteritems():
                    add_poly = False
                    if i == 0:
                        add_poly = True
                    else:
                        if not x_data[i] == x_data[i-1] \
                        or not y_data[i] == y_data[i-1]:
                            add_poly = True

                    if add_poly:
                        q = 0.0
                        if q_name:
                            val = numpy.asarray(self.data[q_name])
                            if len(val.shape) > 1:
                                val = val[i,0]
                            else:
                                val = val[i]
                            q = val

                        # update transformation matrix
                        T = self.T
                        c = numpy.cos(q); s = numpy.sin(q)
                        T[0,0] = c; T[0,1] = -s
                        T[1,0] = s; T[1,1] =  c

                        hull = numpy.asarray(self.data[poly_name][i])
                        hull = numpy.vstack((hull, hull[0,:]))
                        points = numpy.asarray((x_data[i], y_data[i]))

                        # first rotate
                        hull = T.dot(hull.transpose()).transpose()
                        hull = hull + points

                        # is there already a polygon for this index
                        if i in self.bird_view_polys[x_name][poly_name]:
                            poly = self.bird_view_polys[x_name][poly_name][i]
                            if (poly.get_xy != hull).any():
                                poly.set_xy(hull)
                        else:
                            poly = plt.Polygon(hull, **poly_map)
                            self.bird_view_polys[x_name][poly_name][i] = poly
                            self.bird_view_axis.add_patch(poly)

            # add last value to preview plot for blending
            dummy = {x_name : x_data, y_name : y_data}
            for name in (x_name, y_name):
                if name in blend:
                    blend[name].append(dummy[name][-1])

            # after data is assembled add them to plots
            line.set_xdata(x_data)
            line.set_ydata(y_data)

        # PREVIEW
        ## calculate last time increment
        #T = self.data['T'][-1]
        #N = self.data['N'][-1]
#
        ## extend time by one horizon for preview
        #preview_time = numpy.zeros((time.shape[0] + N,))
        #preview_time[:time.shape[0]] = time
        #preview_time[time.shape[0]:] = numpy.arange(0, T*N, T) + T + time[-1]

        blend_subs = {
            'C_kp1_y' : 'c_k_y',
            'C_kp1_x' : 'c_k_x',
            'F_k_y' : 'f_k_y',
            'F_k_x' : 'f_k_x',
            'Z_kp1_y' : 'z_k_y',
            'Z_kp1_x' : 'z_k_x',
        }
        for item in self.preview_mapping:
            # get names from mapping
            x_name = item[0][0]
            y_name = item[1][0]
            q_name = None
            # get theta name only when defined
            if len(item) > 2:
                q_name = item[2][0]

            # get line
            line = self.bird_view_lines[x_name]

            # define data
            x_data = blend.get(blend_subs.get(x_name,''), [])
            y_data = blend.get(blend_subs.get(y_name,''), [])
            q_data = blend.get(blend_subs.get(q_name,''), [])

            # extend lists with current preview data to generate preview
            x_data.extend(self.data[x_name][-1])
            y_data.extend(self.data[y_name][-1])
            if q_name:
                q_data.extend(self.data[q_name][-1])
            else:
                q_data.extend([0]*len(x_data))

            # assemble and transform polygons
            points = numpy.asarray(
                zip(self.data[x_name][-1], self.data[y_name][-1])
            )

            # if we plot foot positions draw also foot hull
            for poly_name, poly_map in self.polygons_mapping.get(x_name, {}).iteritems():
                q = 0.0
                if q_name:
                    val = numpy.asarray(self.data[q_name])
                    if len(val.shape) > 1:
                        val = val[i,0]
                    else:
                        val = val[i]
                    q = val

                # update transformation matrix
                T = self.T
                c = numpy.cos(q); s = numpy.sin(q)
                T[0,0] = c; T[0,1] = -s
                T[1,0] = s; T[1,1] =  c

                # get hull as numpy array
                hull = numpy.asarray(self.data[poly_name][i])
                hull = numpy.vstack((hull, hull[0,:]))

                # iterate over all
                for j in range(points.shape[0]):
                    # first rotate
                    dummy = T.dot(hull.transpose()).transpose()
                    dummy = dummy + points[j]

                    # for preview add dotted linestyle
                    poly_map['ls'] = 'dotted'

                    # is there already a polygon for this index
                    if j in self.bird_view_polys[x_name][poly_name]:
                        poly = self.bird_view_polys[x_name][poly_name][j]
                        if (poly.get_xy != dummy).any():
                            poly.set_xy(dummy)
                    else:
                        poly = plt.Polygon(dummy, **poly_map)
                        self.bird_view_polys[x_name][poly_name][j] = poly
                        self.bird_view_axis.add_patch(poly)

            line.set_xdata(x_data)
            line.set_ydata(y_data)

        # AFTERMATH
        # recalculate x and y limits
        if not self.bird_view_limits:
            self.bird_view_axis.relim()
            self.bird_view_axis.autoscale_view()

        self.bird_view_axis.set_aspect('equal')

        # define legend
        self.bird_view_axis.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))

        # show canvas
        if self.show_canvas:
            # TODO problem of background is not refreshed
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.save_to_file:
            self._save_to_file()

    def create_reference_plot(self):
        """ create plot like that from Maximilien """
        self.reference_fig = plt.figure()
        ax = self.reference_fig.add_subplot(1,1,1)
        ax.set_ylabel("Walking Forward")
        ax.set_xlabel("time [s]")

        # retrieve data from data structure
        time = numpy.asarray(self.data['time'])

        for item in self.bird_view_mapping:
            # get names from mapping
            x_name = item[0][0]; x_map  = item[0][1]
            y_name = item[1][0]; y_map  = item[1][1]
            q_name = None;       q_map = None
            # get theta name only when defined
            if len(item) > 2:
                q_name = item[2][0]; q_map  = item[2][1]
                q_line, = ax.plot([], [], **q_map)

            # get line
            x_line, = ax.plot([], [], **x_map)
            y_line, = ax.plot([], [], **y_map)

            # define data
            x_data = numpy.ones(time.shape[0])*numpy.nan
            y_data = numpy.ones(time.shape[0])*numpy.nan
            q_data = numpy.ones(time.shape[0])*numpy.nan

            for i in range(time.shape[0]):
                # x value conversion
                val = numpy.asarray(self.data[x_name])
                if len(val.shape) > 1:
                    val = val[i,0]
                else:
                    val = val[i]
                x_data[i] = val

                # y value conversion
                val = numpy.asarray(self.data[y_name])
                if len(val.shape) > 1:
                    val = val[i,0]
                else:
                    val = val[i]
                y_data[i] = val

                # optional theta value conversion
                if q_name:
                    val = numpy.asarray(self.data[q_name])
                    if len(val.shape) > 1:
                        val = val[i,0]
                    else:
                        val = val[i]
                    q_data[i] = val

            # plot lines
            x_line.set_xdata(time); x_line.set_ydata(x_data)
            y_line.set_xdata(time); y_line.set_ydata(y_data)
            q_line.set_xdata(time); q_line.set_ydata(y_data)


        # recalculate x and y limits
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal')

        # define legend position
        # Put a legend to the right of the current axis
        legend = ax.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))

        # show canvas
        if self.show_canvas:
            self.reference_fig.show()
            plt.pause(1e-8)

        if self.save_to_file:
            self._save_to_file(figure=self.reference_fig)

    def create_data_plot(self):
        """ create plot of problem data """
        # CPUTIME
        self.data_cpu_fig  = plt.figure()
        ax  = self.data_cpu_fig.add_subplot(1,1,1)
        ax.set_title('CPU Time of Solvers')
        ax.set_ylabel("CPU time [ms]")
        ax.set_xlabel("no. of iterations")

        # retrieve data from data structure
        if 'ori_qp_cputime' in self.data:
            ori_cpu = numpy.asarray(self.data['ori_qp_cputime'])
            pos_cpu = numpy.asarray(self.data['pos_qp_cputime'])
            idx = numpy.asarray(range(len(ori_cpu)))

            # get bar plots
            width = 0.3
            ori_bar = ax.bar(idx, ori_cpu, width, linewidth=0, color='r')
            pos_bar = ax.bar(idx, pos_cpu, width, linewidth=0, color='y',
                bottom=ori_cpu
            )

            # define legend position
            # Put a legend to the right of the current axis
            legend = ax.legend(
                (ori_bar[0], pos_bar[0]), ('$QP_{\\theta}$', '$QP_{x,y}$')
            )
        else:
            qp_cpu = numpy.asarray(self.data['qp_cputime'])
            idx = numpy.asarray(range(len(qp_cpu)))

            # get bar plots
            width = 0.3
            qp_bar = ax.bar(idx, qp_cpu, width, linewidth=0, color='g')

            # define legend position
            # Put a legend to the right of the current axis
            legend = ax.legend(
                (qp_bar[0],), ('$QP_{x,y,\\theta}$',)
            )

        # recalculate x and y limits
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=True, tight='True')
        #ax.set_aspect('equal')

        # NWSR
        self.data_nwsr_fig = plt.figure()
        ax  = self.data_nwsr_fig.add_subplot(1,1,1)
        ax.set_title('Working Set Recalculation of Solvers')
        ax.set_ylabel("no. of WSR")
        ax.set_xlabel("no. of iterations")

        # retrieve data from data structure
        if 'ori_qp_nwsr' in self.data:
            ori_nwsr = numpy.asarray(self.data['ori_qp_nwsr'])
            pos_nwsr = numpy.asarray(self.data['pos_qp_nwsr'])
            idx = numpy.asarray(range(len(ori_cpu)))

            # get bar plots
            width = 0.3
            ori_bar = ax.bar(idx, ori_nwsr, width, linewidth=0, color='r')
            pos_bar = ax.bar(idx, pos_nwsr, width, linewidth=0, color='y',
                bottom=ori_nwsr
            )

            # define legend position
            # Put a legend to the right of the current axis
            legend = ax.legend(
                (ori_bar[0], pos_bar[0]), ('$QP_{\\theta}$', '$QP_{x,y}$')
            )
        else:
            qp_nwsr = numpy.asarray(self.data['qp_nwsr'])
            idx = numpy.asarray(range(len(qp_nwsr)))

            # get bar plots
            width = 0.3
            qp_bar = ax.bar(idx, qp_nwsr, width, linewidth=0, color='g')

            # define legend position
            # Put a legend to the right of the current axis
            legend = ax.legend(
                (qp_bar[0],), ('$QP_{x,y,\\theta}$',)
            )

        # recalculate x and y limits
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=True, tight='True')
        #ax.set_aspect('equal')

        # show canvas
        if self.show_canvas:
            self.data_cpu_fig.tight_layout()
            self.data_cpu_fig.show()
            self.data_nwsr_fig.tight_layout()
            self.data_nwsr_fig.show()
            plt.pause(1e-8)

        if self.save_to_file:
            self._save_to_file(figure=self.data_cpu_fig)
            self._save_to_file(figure=self.data_nwsr_fig)



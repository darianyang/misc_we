#!/usr/bin/env python
# By Alex DeGrave, July 16 2015
import argparse
import h5py
import math
import matplotlib
matplotlib.use('Agg')
import numpy
import pylab
import scipy.ndimage
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
Class to plot contour maps of free energy. See --help for options.
'''
def get_parents(walker_tuple, h5_file):
    it, wlk = walker_tuple
    parent = h5_file["iterations/iter_{:08d}".format(it)]["seg_index"]["parent_id"][wlk]
    return it-1, parent

def trace_walker(walker_tuple, h5_file):
    # Unroll the tuple into iteration/walker 
    it, wlk = walker_tuple
    # Initialize our path
    path = [(it,wlk)]
    # And trace it
    while it > 1: 
        it, wlk = get_parents((it, wlk), h5_file)
        path.append((it,wlk))
    return numpy.array(sorted(path, key=lambda x: x[0]))

def get_pcoords(path, h5_file):
    # Initialize a list for the pcoords
    pcoords = []
    # Loop over the path and get the pcoords for each walker
    for it, wlk in path:
        # Here we are taking every 10 time points, feel free to adjust to see what that does
        pcoords.append(h5_file['iterations/iter_{:08d}'.format(it)]['pcoord'][wlk][::10,:])
    return numpy.array(pcoords)

def adjust_plot():
    import matplotlib.pyplot as pyplot
    import h5py

    # Now we can play with some fun stuff
    # First we'll pull which iteration we are on using the 
    # file created by our movie loop
    cur_iter = 150

    # Now we want to pull our tranjectory and  plot it 
    # until the current iteration. If you have long 
    # trajectories or a lot of data it's a good idea
    # to calculate this beforehand and just open a pickled 
    # file or something instead of re-calculating it every 
    # iteration like it's done here.
    with h5py.File("west.h5", "r") as w:
        # We will trace a specific trajectory from the latest 
        # iteration to the beginning and then use only the part 
        # that we currently need
        #last_iter = w.attrs['west_current_iteration'] - 1
        last_iter = 150
        # Say we pull the 0th walker and trace that 
        #path2 = trace_walker((last_iter, 2), w)
        #path = trace_walker((last_iter, 254), w)
        #print path
        # And pull pcoords for the path calculated
        #pcoords2 = get_pcoords(path2, w)
        #pcoords = get_pcoords(path, w)
        #print pcoords[:,-1]
        #print pcoords.shape
        #cur_pcoords2 = pcoords2[:cur_iter,:,:]
        pcoordx = numpy.loadtxt('trace.txt', usecols=5)
        pcoordy = numpy.loadtxt('trace.txt', usecols=6)
        #cur_pcoords = pcoords[:cur_iter,:,:]
        #shp = cur_pcoords.shape
        #shp2 = cur_pcoords2.shape
        #cur_pcoords = cur_pcoords.reshape(shp[0]*shp[1], shp[2])
        #cur_pcoords2 = cur_pcoords2.reshape(shp2[0]*shp2[1], shp2[2])
        # We got the full path, now plot up to the part we need
    #    numpy.savetxt("pcoords.txt", cur_pcoords)
        #print cur_pcoords.shape
        #print cur_pcoords[:1000:2]
        #print cur_pcoords2[:869:4]
        pyplot.plot(pcoordx[::], pcoordy[::], c="gold", lw=5)#marker="o", markersize=30, lw=4)
        #pyplot.plot(cur_pcoords2[:869:4,0], cur_pcoords2[:869:4,1], '--', c="dodgerblue", lw=5)#marker="^", markersize=30, lw=4)
        #pyplot.plot(cur_pcoords[:1000,0], cur_pcoords[:1000,1], c="aqua", lw=2)
        xs = [1.6033, 3.2284, 4.3233, 5.0776]
        ys = [2.7923, 3.0541, 3.689, 3.5795]
#        pyplot.scatter(xs[0], ys[0], s=5000, marker='*', linewidth=2, edgecolor='k', c="gold", alpha=1, zorder=3, clip_on=False)
        #pyplot.scatter(xs[3], ys[3], s=1000, marker='o', linewidth=2, edgecolor='k', c="gold", alpha=1, zorder=3)
        pyplot.scatter(xs, ys, s=1000, marker='o', linewidth=2, edgecolor='k', c="red", alpha=1, zorder=3, clip_on=False)
        #pyplot.scatter(xs[2], ys[2], s=5000, marker='o', linewidth=2, edgecolor='k', c="green", alpha=1, zorder=3, clip_on=False)
        # Say we pull the 300th walker and trace that 
#        path = trace_walker((last_iter, 326), w)
#        # And pull pcoords for the path calculated
#        pcoords = get_pcoords(path, w)
#        cur_pcoords = pcoords[:33,:,:]
#        shp = cur_pcoords.shape
#        cur_pcoords = cur_pcoords.reshape(shp[0]*shp[1], shp[2])
#        print("Current pcoords shape")
#        print(cur_pcoords.shape)
#        # We got the full path, now plot up to the part we need
#        #import IPython
#        #IPython.embed()
#        pyplot.plot(cur_pcoords[:,0], cur_pcoords[:,1], c="black", lw=3)
#        pyplot.plot(cur_pcoords[:,0], cur_pcoords[:,1], c="cyan", lw=2)

class ContourPlotter:
    def __init__(self):
        self._parse_args()

    def _parse_args(self):
        '''
        Parse command line arguments.
        '''
        parser = argparse.ArgumentParser()

        # Data input options
        parser.add_argument('--text-input',
                            dest='text_file',
                            help='Load data from TEXT_FILE. '
                                 'The file should be ``numpy.loadtxt``-able.',
                            type=str)
        parser.add_argument('--pdist-input', default=None,
                          dest='pdist_file',
                          help='Input from the specified w_pdist output file '
                               '(HDF5 format). '
                               'Use with ``--pdist-axes`` (mandatory). '
                               'Optionally, also include ``--first-iter`` and '
                               '``--last-iter`` flags.',
                          type=str)
        parser.add_argument('--text-hist', default=None,
                            dest='text_hist',
                            help='Load data from TEXT_HIST. '
                                 'The file should be ``numpy.loadtxt``-able',
                            type=str)
        parser.add_argument('--pdist-axes', default="(0,1)",
                          dest='pdist_axes',
                          help='Plot PDIST_AXES of the w_pdist file specified '
                               'with ``--pdist-input`` PDIST_AXES should be a '
                               'string that can be parsed as a Python tuple '
                               'or list with two elements (for example, '
                               '``(0,1)``. Use with ``--pdist-input``.',
                          type=str)
        parser.add_argument('--first-iter', default=None,
                          dest='first_iter',
                          help='Plot data starting at iteration FIRST_ITER. '
                               'By default, plot data starting at the first ' 
                               'iteration in the specified w_pdist file. '
                               'Use with ``--pdist-input``.',
                          type=int)
        parser.add_argument('--last-iter', default=None,
                          dest='last_iter',
                          help='Plot data up to and including iteration '
                               'LAST_ITER. By default, plot data up to and '
                               'including the last iteration in the specified '
                               'w_pdist file.  Use with ``--pdist-input``.',
                          type=int)

        # Binning options
        parser.add_argument('--xbins', default=100,
                            dest='xbins',
                            help='Use XBINS number of bins for histogramming '
                                 'along the x-axis. Divide the range between '
                                 'the minimum and maximum observed x-value '
                                 'into this many bins',
                            type=int)
        parser.add_argument('--xbinexpr', default=None,
                            dest='xbinexpr',
                            help='Parse XBINEXPR as a list and use it as a '
                                 'list of bin edges along the x-axis.',
                            type=str)
        parser.add_argument('--ybins', default=100,
                            dest='ybins',
                            help='Use YBINS number of bins for histogramming ' 
                                 'along the y-axis.  Divide the range between '
                                 'the minimum and maxium observed y-value '
                                 'into this many bins.',
                            type=int)
        parser.add_argument('--ybinexpr', default=None,
                            dest='ybinexpr',
                            help='Parse YBINEXPR as a list and use it as a '
                                 'list of bin edges along the y-axis.',
                            type=str)
        parser.add_argument('--zbins', default=10,
                            dest='zbins',
                            help='Use ZBINS number of bins for histogramming' 
                                 'along the z-axis.  Divide the range between '
                                 'ZMIN and ZMAX (or the minimum and maxium '
                                 'observed z-value into this many bins.',
                            type=int)
        parser.add_argument('--zbinexpr', default=None,
                            dest='zbinexpr',
                            help='Parse ZBINEXPR as a list and use it as a '
                                 'list of bin edges along the z-axis.',
                            type=str)
        parser.add_argument('--zmax', default=None,
                            dest='zmax',
                            help='Use ZMAX as the maximum value to plot along '
                                 'the z-axis (that is, the color scale). By '
                                 'default, use the maximum observed value. The ' 
                                 '``--zbinexpr`` option will override this '
                                 'option.',
                            type=float)
        parser.add_argument('--zmin', default=None,
                            dest='zmin',
                            help='Use ZMIN as the minimum value to plot along '
                                 'the z-axis (that is, the color scale). By '
                                 'default, use the minimum observed value. The '
                                 '``--zbinexpr`` option will override this '
                                 'option.',
                            type=float)
        parser.add_argument('--Q-x', default=None,
                            dest='xcontact_count',
                            help='By specifying an integer for XCONTACT_COUNT,'
                                 ' this setting will automatically generate '
                                 'bins along the x-axis.  This assumes that '
                                 'the x-axis contains fraction of native '
                                 'contacts data, and that a total of '
                                 'YCONTACT_COUNT native contacts are possible.'
                                 ' Uses one bin per native contact.',
                            type=int)
        parser.add_argument('--Q-y', default=None,
                            dest='ycontact_count',
                            help='By specifying an integer for YCONTACT_COUNT,'
                                 ' this setting will automatically generate '
                                 'bins along the y-axis.  This assumes that '
                                 'the y-axis contains fraction of native '
                                 'contacts data, and that a total of '
                                 'YCONTACT_COUNT native contacts are possible.'
                                 ' Uses one bin per native contact.',
                            type=int)

        # Plot label and general output options
        parser.add_argument('--output', default='hist.pdf',
                            dest='output_path',
                            help='The filename to which the plot will be saved.'
                                 ' Various image formats are available.  You ' 
                                 'may choose one by specifying an extension',
                            type=str)
        parser.add_argument('--title', default=None,
                            dest='title',
                            help='Add ``TITLE`` as a title for the plot.',
                            type=str)
        parser.add_argument('--xlabel', default=None,
                            dest='xlabel',
                            help='Use XLABEL as a label for the x-axis, '
                                 'where the x-axis with be the zero-th '
                                 'dimension the dataset you supply.',
                            type=str)
        parser.add_argument('--ylabel', default=None,
                            dest='ylabel',
                            help='Use YLABEL as a label for the y-axis, '
                                 'where the y-axis with be the first '
                                 'dimension the dataset you supply.',
                            type=str)
        parser.add_argument('--xrange', default=None,
                            dest='xplotrange',
                            help='Extend or restrict the x-axis to include '
                                 'values in XPLOTRANGE. XPLOTRANGE should '
                                 'be a str that Python can parse as a tuple.'
                            )
        parser.add_argument('--yrange', default=None,
                            dest='yplotrange',
                            help='Extend or restrict the y-axis to include '
                                 'values in YPLOTRANGE. YPLOTRANGE should '
                                 'be a str that Python can parse as a tuple.'
                            )


        # Plot style and data smoothing options
        parser.add_argument('--cmap', default='magma_r',
                            dest='cmap',
                            help='Use the colormap CMAP for the z-axis of the ' 
                                 'plot. Ex: ``hot``,``jet``,``RdBu``. '
                                 'See MatPlotLib documentation for more '
                                 'options.',
                            type=str)
        parser.add_argument('--smooth-data', default = None, 
                            dest='data_smoothing_level',
                            help='Smooth data (plotted as histogram or contour'
                                 ' levels) using a gaussian filter with sigma='
                                 'DATA_SMOOTHING_LEVEL.',
                            type=float)
        parser.add_argument('--smooth-curves', default = None, 
                            dest='curve_smoothing_level',
                            help='Smooth curves (plotted as lines between '
                                 'levels) using a gaussian filter with sigma='
                                 'CURVE_SMOOTHING_LEVEL.',
                            type=float)
        parser.add_argument('--plot-mode', default = 'contourf_l',
                            dest='plot_mode',
                            help='Use plotting mode PLOT_MODE. Options are: '
                                 '``contourf``--plot contour levels. '
                                 '``histogram``--plot histogram. '
                                 '``lines``--plot contour lines only. '
                                 '``contourf_l``--plot contour levels and lines. '
                                 '``histogram_l``--plot histogram and contour lines. ',
                            type=str)
        parser.add_argument('--postprocess', default = None,
                            dest='postprocess_func',
                            help='After plotting data, load and execute the '
                                 'Python function specified by '
                                 'POSTPROCESS_FUNC. POSTPROCESS_FUNC should be '
                                 'a string of the form ``mymodule.myfunction``.'
                                 'Example: '
                                 '``--postprocess mymodule.myfunction``.'
                            )     
        self.args = parser.parse_args()

        # Figure out what kind of input to use.
        self.mode = None
        if self.args.pdist_file is not None: # Use w_pdist input.
            if self.mode is None:
                self.mode = 'pdist'
            else: 
                print('You specified multiple input types.  Please only '
                      ' specify one type.')
        if self.args.text_file is not None: # Use text data input.
            if self.mode is None:
                self.mode = 'text'
            else: 
                print('You specified multiple input types.  Please only '
                      ' specify one type.')
        if self.args.text_hist is not None: # Use text histogram input.
            if self.mode is None:
                self.mode = 'text_hist'
            else: 
                print('You specified multiple input types.  Please only '
                      ' specify one type.')

        # Figure out the plotting mode
        if self.args.plot_mode not in ['contourf',
                                       'histogram',
                                       'lines',
                                       'contourf_l',
                                       'histogram_l']:
            print('Plot mode ``%s`` is not a valid option. See ``--help`` for '
                  'more information.' % self.args.plot_mode)
            exit(1)
        else:
            self.plot_mode = self.args.plot_mode

        self.data_smoothing_level = self.args.data_smoothing_level
        self.curve_smoothing_level = self.args.curve_smoothing_level

        if self.args.xplotrange is not None:
            self.xplotrange = self._get_bins_from_expr(self.args.xplotrange)
        if self.args.yplotrange is not None:
            self.yplotrange = self._get_bins_from_expr(self.args.yplotrange)


    def _get_bins_from_expr(self, binexpr):
        '''
        Convert ``binexpr`` from a string to a python iteratable.  This method
        makes the ``numpy`` and ``math`` libraries available for evaluation of
        the string.  Use ``inf`` to access ``float('inf')``. 
        '''
        self.binexpr_namespace = {'numpy': numpy,
                                  'math':math,
                                  'inf':float('inf')}
        bins_ = eval(binexpr,self.binexpr_namespace)
        if float('inf') in bins_:
            print("Warning! One of the bin boundaries is 'inf' (infinity). " 
                  "Attempting to make a normalized histogram with a bin " 
                  "boundary at infinity does NOT make sense. It only makes "
                  "sense to use 'inf' when specifying bin boundaries for the "
                  "z-axis.")
        return bins_

    def _make_bin_bounds(self):
        '''
        Set the attributes ``xedges`` and ``yedges`` by parsing the command line
        arguments ``xbins``, ``ybins``, ``xbinexpr``, ``ybinexpr``, 
        ``xcontact_count``, and ``ycontact_count``. If the user specified 
        ``xbinexpr`` or ``ybinexpr``, these take precedence over ``xbins`` and 
        ``ybins``. Similarly, ``xcontact_count`` and ``ycontact_count`` take 
        precedence over ``xbinexpr`` and ``ybinexpr``. If ``--pdist-input`` is
        specified, the ``xcontact_count``, ``ycontact_count``, ``xbinexpr``, 
        and ``ybinexpr`` arguments take precedence over the bin boundaries
        saved in the pdist file.  If these arguments are not specified, then 
        this method defaults to using the saved bin boundaries.
        '''
        # X
        if self.args.xcontact_count is not None:
            delta = 1./self.args.xcontact_count
            self.xedges=numpy.arange(-delta/2,1+delta,delta)
        elif self.args.xbinexpr is not None:
            self.xedges = self._get_bins_from_expr(self.args.xbinexpr)
        elif self.mode == 'pdist':
            # Save the binbounds as attributes. 
            self.xedges = numpy.array(self.pdist_HDF5['binbounds_%d'%self.axis_list[0]])
            self.xedges = numpy.array(self.pdist_HDF5['binbounds_%d'%self.axis_list[0]])
        else:
            minx = self.data[:,0].min()
            maxx = self.data[:,0].max()
            r = maxx - minx
            delta = r/self.args.xbins
            self.xedges = numpy.arange(minx,maxx+r,r)

        # Y
        if self.args.ycontact_count is not None:
            delta = 1./self.args.ycontact_count
            self.yedges=numpy.arange(-delta/2,1+delta,delta)
        elif self.args.ybinexpr is not None:
            self.yedges = self._get_bins_from_expr(self.args.ybinexpr)
        elif self.mode == 'pdist':
             # Save the binounds as attributes. 
             self.yedges = numpy.array(self.pdist_HDF5['binbounds_%d'%self.axis_list[1]])
             self.yedges = numpy.array(self.pdist_HDF5['binbounds_%d'%self.axis_list[1]])
        else:
            miny = self.data[:,0].min()
            maxy = self.data[:,0].max()
            r = maxy - miny
            delta = r/self.args.ybins
            self.yedges = numpy.arange(miny,maxy+r,r)

    def _load_from_text_file(self):
        '''
        Load data from a text file.  This is just a wrapper for the 
        ``loadtxt`` method of the main numpy module.
        '''
        self.data = numpy.loadtxt(self.args.text_file)

    def _sum_except_along(self,array,axis_list):
        '''
        Sum along along all dimensions of ``array`` except those specified in 
        the ``axis_list``, which should be a tuple (or other iteratable) of 
        integers. Return the summed array.
        '''
        axis_list = numpy.array(axis_list)
        # Convert axis indices to the indexing scheme used in the pdist 
        # histogram.  Add one, as axis zero is the iteration.
        axis_list += 1
        a = array
        for i in range(len(array.shape)):
            if i not in axis_list:
                a = a.sum(axis=i)
        return a

    def _load_from_pdist_file(self):
        '''
        Load data from a w_pdist output file. This includes bin boundaries. 
        '''
        # Open the HDF5 file.
        self.pdist_HDF5 = h5py.File(self.args.pdist_file)

        # Load the histograms and sum along all axes except those specified by
        # the user.  Also, only include the iterations specified by the user.
        histogram      = numpy.array(self.pdist_HDF5['histograms'])

        # Figure out what iterations to use
        n_iter_array   = numpy.array(self.pdist_HDF5['n_iter'])
        if self.args.first_iter is not None:
            first_iter = self.args.first_iter
        else:
            first_iter = n_iter_array[0] 
        if self.args.last_iter is not None:
            last_iter = self.args.last_iter
        else:
            last_iter = n_iter_array[-1]
        first_iter_idx = numpy.where(n_iter_array == first_iter)[0][0]
        last_iter_idx  = numpy.where(n_iter_array == last_iter)[0][0]
        histogram      = histogram[first_iter_idx:last_iter_idx+1]

        # Sum along axes
        self.axis_list = self._get_bins_from_expr(self.args.pdist_axes)
        self.H         = self._sum_except_along(histogram, self.axis_list) 

        # Make sure that the axis ordering is correct.
        if self.axis_list[0] > self.axis_list[1]:
            self.H = self.H.transpose()



    def _load_from_plothist_HDF5(self):
        '''
        Load data from an HDF5 file.  This method assumes the HDF5 file 
        structure is the same as the WESTPA tool ``plothist`` would output.

        Currently, this method is only a placeholder.
        '''
        pass

    def _histogram(self):
        '''
        Call ``_make_bin_bounds`` to generate bin bounds, and then use these
        bins to histogram the data in ``self.data``.  Normalize the data so 
        that the histogram is an estimate of an underlying probability density
        function (the histogram should integrate to 1). Save the resulting data
        in the attribute ``self.H`` (a histogram).
        '''
        self._make_bin_bounds()
        H, xedges, yedges = numpy.histogram2d(self.data[:,0], self.data[:,1], 
                                              bins=[self.xedges,self.yedges],
                                              normed=True)
        self.H = H

    def _edges_to_midpoints(self):
        '''
        Get the midpoints of bins stored in ``self.xedges`` and ``self.yedges``.
        Store midpoints in ``self.x_mids`` and ``self.y_mids``.
        '''
        self.x_mids = [(self.xedges[i]+self.xedges[i+1])/2\
                       for i in range(self.xedges.shape[0] -1)]
        self.y_mids = [(self.yedges[i]+self.yedges[i+1])/2\
                       for i in range(self.yedges.shape[0] -1)]

    def _make_Z_binbounds(self):
        ''' 
        Set the attribute ``self.zedges`` by figuring out the bin boundaries to
        use along the z-axis.  Use the bin expression that the user specified
        in ``self.args.zbinexpr``, if it is available.  If not, figure out the 
        range of z-values to plot.  If either is specified, maximum or minimum 
        value (or both) come from ``self.args.zmax`` and ``self.args.zmin``. 
        For a maximum or minimum value not specified, use the maximum or 
        minimum value observed in self.Z.  Finally, divide this range into the
        number of bins the user specified in ``self.args.zbins``.
        ''' 
        if self.args.zbinexpr is not None:
            self.zedges = self._get_bins_from_expr(self.args.zbinexpr) 
        else:
            if self.args.zmin is not None:
                self.zmin = self.args.zmin
            else:
                self.zmin = numpy.nanmin(self.Z)
            if self.args.zmax is not None:
                self.zmax = self.args.zmax
            else:
                self.zmax = numpy.nanmax(self.Z)
            nbins  = self.args.zbins
            range_ = self.zmax - self.zmin
            delta  = range_/nbins
            self.zedges = numpy.arange(self.zmin, self.zmax+delta, delta)

    def _smooth(self):
        if self.data_smoothing_level is not None:
            self.Z_data[numpy.isnan(self.Z_data)] = numpy.nanmax(self.Z_data)
            self.Z_data = scipy.ndimage.filters.gaussian_filter(self.Z_data, 
                              self.data_smoothing_level)
        if self.curve_smoothing_level is not None:
            self.Z_curves[numpy.isnan(self.Z_curves)] = numpy.nanmax(self.Z_curves)
            self.Z_curves = scipy.ndimage.filters.gaussian_filter(self.Z_curves, 
                                self.curve_smoothing_level)
        self.Z_data[numpy.isnan(self.Z)] = numpy.nan 
        self.Z_curves[numpy.isnan(self.Z)] = numpy.nan 

    def _do_contourf(self):
        '''
        Plot contour levels only. 
        '''
        ax = self.ax
        X = self.x_mids
        Y = self.y_mids

        # Take care of 'nan' values
        self.Z_data[numpy.isnan(self.Z_data)] = numpy.nanmax(self.Z_data)

        # Plot contours.
        p = ax.contourf(X, Y, self.Z_data.T, self.zedges, 
                        cmap=pylab.cm.get_cmap(self.cmap, len(self.zedges)-1)
                        )
        # Add a colorbar
        self.cbar = ax.colorbar()

    def _do_contourf_l(self):
        '''
        Plot contour levels with black lines between then. 
        '''
        ax = self.ax
        X = self.x_mids
        Y = self.y_mids
        # Take care of 'nan' values
        self.Z_data[numpy.isnan(self.Z_data)] = max(self.zedges)*2
        self.Z_curves[numpy.isnan(self.Z_curves)] = max(self.zedges)*2

        # Plot contours.
        p = pylab.contourf(X, Y, self.Z_data.T, self.zedges, 
                        cmap=pylab.cm.get_cmap(self.cmap, len(self.zedges)-1)
                        )
        # Add a colorbar
        self.cbar = self.fig.colorbar(p, orientation='vertical', ticks=[0,2,4,6,8,10,12,14,16], fraction=0.044, pad=0.05)#,ticks=[]

        # Plot contour lines.
        p = ax.contour(X, Y, self.Z_curves.T, self.zedges, 
                       colors='k') 

    def _do_lines(self):
        '''
        Plot contour lines only. 
        '''
        ax = self.ax
        X = self.x_mids
        Y = self.y_mids

        # Take care of 'nan' values
        self.Z_curves[numpy.isnan(self.Z_curves)] = max(self.zedges)*2

        # Plot contour lines.
        p = ax.contour(X, Y, self.Z_curves.T, self.zedges, 
                       cmap=pylab.cm.get_cmap(self.cmap, len(self.zedges)-1)
                       )
        # Add a colorbar
        self.cbar = self.fig.colorbar(p)

    def _do_histogram(self): 
        '''
        Plot histogram only.
        '''
        ax = self.ax
        X = self.xedges
        Y = self.yedges
        Zm = ma.array(self.Z_data, mask=numpy.isnan(self.Z_data))
        p = ax.pcolormesh(X, Y, Zm.T, 
                          cmap=self.cmap,
                          vmin=self.zedges[0],
                          vmax=self.zedges[-1])
        p.set_edgecolor('face')

        # Add a colorbar
        self.cbar = self.fig.colorbar(p)

    def _do_histogram_l(self): 
        '''
        Plot histogram with contour lines. 
        '''
        ax = self.ax
        X = self.xedges
        Y = self.yedges
        Zm = ma.array(self.Z_data, mask=numpy.isnan(self.Z_data))
        p = ax.pcolormesh(X, Y, Zm.T, 
                          cmap=self.cmap,
                          vmin=self.zedges[0],
                          vmax=self.zedges[-1])

        # Get rid of lines between cells
        p.set_edgecolor('face')

        # Add a colorbar
        self.cbar = self.fig.colorbar(p)

        X = self.x_mids
        Y = self.y_mids
        p = ax.contour(X, Y, self.Z_curves.T, self.zedges, 
                       colors='k') 

    def _make_plot(self):
        '''
        Make a contour plot/histogram of free energy using the probabilities 
        stored in ``self.H``. 
        '''
        matplotlib.rcParams['font.size'] = 25

        # Convert to free energy
        self.Z = -1*numpy.log(self.H)

        # Make everything relative to minimum, at zero.
        self.Z -= numpy.nanmin(self.Z)

        # Take care of 'inf' values.
        self.Z[self.H==0] = numpy.nan 

        # Do data smoothing. We have to make copies of the array so that
        # the data and curves can have different smoothing levels.
        self.Z_data = numpy.copy(self.Z)
        self.Z_curves = numpy.copy(self.Z)
        self._smooth()

        # Calculate edges along z-axis
        self._make_Z_binbounds()

        # Specify a color map.
        self.cmap = pylab.cm.cmap_d[self.args.cmap]
        self.cmap.set_bad(color='white')
        self.cmap.set_over(color='white')
        self.cmap.set_under(color='white')

        # Make the figure and axes
        self.fig = pylab.figure(figsize=(24.2,20))
        #self.fig.text(0.20,0.70,'Reactants', fontsize=75)
        #self.fig.text(0.55,0.24,'Products', fontsize=75)
        self.ax = self.fig.add_subplot(1,1,1)
        
        if self.plot_mode == 'contourf':
            self._do_contourf()
        if self.plot_mode == 'contourf_l':
            self._do_contourf_l()
        if self.plot_mode == 'lines':
            self._do_lines()
        if self.plot_mode == 'histogram':
            self._do_histogram()
        if self.plot_mode == 'histogram_l':
            self._do_histogram_l()


    def _format_plot(self):
        '''
        Add axis labels and a plot title, if the user specified them.
        Also, change the bounds of the plot to match the minimum and maximum
        values for the bin boundaries along the x- and y-axis.
        '''
        if self.args.xlabel is not None:
            pylab.xlabel(self.args.xlabel, fontsize=70)
        if self.args.ylabel is not None:
            pylab.ylabel(self.args.ylabel, fontsize=70)
        if self.args.title is not None:
            pylab.title(self.args.title)
        if self.args.xplotrange is not None:
            pylab.xlim(self.xplotrange[0], self.xplotrange[1])
        elif self.args.xcontact_count is None:
            pylab.xlim(self.xedges[0],self.xedges[-1])
        else:
            pylab.xlim(0,1)
        pylab.xticks([2,3,4,5,6], fontsize=55)
        self.ax.tick_params(length=20,width=2)
        pylab.yticks([2,3,4,5,6],fontsize=55)
        if self.args.yplotrange is not None:
            pylab.ylim(self.yplotrange[0], self.yplotrange[1])
        elif self.args.ycontact_count is None:
            pylab.ylim(self.yedges[0],self.yedges[-1])
        else:
            pylab.ylim(0,1)

        # Add a label to the colorbar.
        

        label = r'$\Delta F(x)\,/\,kT$' +'\n' + r'$\left[-\ln\,P(x)\right]$'
       
        self.cbar.set_label(label, fontsize=50, labelpad=50)#, rotation=0)
        #self.cbar.ax.invert_yaxis()
        self.cbar.ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14, 16], fontsize=50)

        # Remove the top and bottom spines
        #for kw in ('top', 'right'):
        #    self.ax.spines[kw].set_visible(False)
        #self.ax.xaxis.set_ticks_position('bottom')
        #self.ax.yaxis.set_ticks_position('left')
        #
        ## Adjust linewidths
        #for kw in ('left', 'bottom'):
        #    self.ax.spines[kw].set_linewidth(1.5)
        #self.ax.tick_params(width=1.5, direction='out')

        #self.fig.subplots_adjust(bottom=0.1, left=0.1)
        #self.ax.set_aspect('equal', 'box')

    def _run_postprocessing(self):
        '''
        Run the user-specified postprocessing function.
        '''
        import importlib
        # Parse the user-specifed string for the module and class/function name.
        module_name, attr_name = self.args.postprocess_func.split('.', 1) 
        # import the module ``module_name`` and make the function/class 
        # accessible as ``attr``.
        attr = getattr(importlib.import_module(module_name), attr_name) 
        # Call ``attr``.
        attr()










































































    def run(self):
        '''
        Main public method of the ContourPlotter class. Call this method to 
        load data, histogram the data, plot a contour plot, and save the 
        contour plot to a file.
        '''
        if self.mode == 'text':
            self._load_from_text_file()
            self._histogram()
            self._edges_to_midpoints()
        elif self.mode == 'text_hist':
            self.H = numpy.loadtxt(self.args.text_hist)
            self._make_bin_bounds()
            self._edges_to_midpoints()
        elif self.mode == 'pdist':
            self._load_from_pdist_file()
            self._make_bin_bounds()
            self._edges_to_midpoints()
        self._make_plot()
        self._format_plot()
        adjust_plot()
        if self.args.postprocess_func is not None:
            self._run_postprocessing()
        pylab.tight_layout()
        pylab.savefig(self.args.output_path, dpi=600)

if __name__ == "__main__":
    contourplotter = ContourPlotter()
    contourplotter.run()

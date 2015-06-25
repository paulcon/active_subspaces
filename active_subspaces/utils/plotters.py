"""Utilities for plotting quantities computed in active subspaces."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, Delaunay, convex_hull_plot_2d, delaunay_plot_2d
import os

def plot_opts(savefigs=True, figtype='.eps'):
    """
    A few options for the plots.

    :param bool savefigs: Save figures into a separate figs director.
    :param str figtype: A file extention for the type of image to save.

    :return: opts, The chosen options. The keys in the dictionary are
        `figtype`, `savefigs`, and `font`. The `font` is a dictionary that
        sets the font properties of the figures.
    :rtype: dict
    """

    # make figs directory
    if savefigs:
        if not os.path.isdir('figs'):
            os.mkdir('figs')

    # set plot fonts
    myfont = {'family' : 'arial',
            'weight' : 'normal',
            'size' : 14}

    opts = {'figtype' : figtype,
            'savefigs' : savefigs,
            'myfont' : myfont}

    return opts

def eigenvalues(e, e_br=None, out_label=None, opts=None):
    """
    Plot the eigenvalues for the active subspace analysis with optional
    bootstrap ranges.

    :param ndarray e: k-by-1 matrix that contains the estimated eigenvalues.
    :param ndarray e_br: Lower and upper bounds for the estimated eigenvalues.
        These are typically computed with a bootstrap.
    :param str out_label: A label for the quantity of interest.
    :param dict opts: A dictionary with some plot options.

    **See Also**

    utils.plotters.eigenvectors
    utils.plotters.subspace_errors
    """
    if opts == None:
        opts = plot_opts()

    k = e.shape[0]
    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=(7,7))
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1 ,k+1), e, 'ko-',markersize=12,linewidth=2)
    if e_br is not None:
        plt.fill_between(range(1, k+1), e_br[:,0], e_br[:,1],
            facecolor='0.7', interpolate=True)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalues')
    plt.title(out_label)
    plt.grid(True)
    plt.xticks(range(1, k+1))
    if e_br is None:
        plt.axis([0, k+1, 0.1*np.amin(e), 10*np.amax(e)])
    else:
        plt.axis([0, k+1, 0.1*np.amin(e_br[:,0]), 10*np.amax(e_br[:,1])])

    if opts['savefigs']:
        figname = 'figs/evals_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    show_plot(plt)
def subspace_errors(sub_br ,out_label=None, opts=None):
    """
    Plot the estimated subspace errors for the active subspace analysis with
    bootstrap ranges.

    :param ndarray sub_br: (k-1)-by-3 matix that contains the lower bound, mean,
        and upper bound of the subspace errors for each dimension of subspace.
    :param str out_label: A label for the quantity of interest.
    :param dict opts: A dictionary with some plot options.

    **See Also**

    utils.plotters.eigenvectors
    utils.plotters.eigenvalues
    """
    if opts == None:
        opts = plot_opts()

    kk = sub_br.shape[0]
    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=(7,7))
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1, kk+1), sub_br[:,1], 'ko-', markersize=12)
    plt.fill_between(range(1, kk+1), sub_br[:,0], sub_br[:,2],
        facecolor='0.7', interpolate=True)
    plt.xlabel('Subspace dimension')
    plt.ylabel('Subspace distance')
    plt.title(out_label)
    plt.grid(True)
    plt.xticks(range(1, kk+1))
    plt.axis([0, kk+1, 0.1*np.amin(sub_br[:,0]), 1])

    if opts['savefigs']:
        figname = 'figs/subspace_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    show_plot(plt)
def eigenvectors(W, W_br=None, in_labels=None, out_label=None, opts=None):
    """
    Plot the estimated eigenvectors for the active subspace analysis with
    optional bootstrap ranges.

    :param ndarray W: m-by-k matrix that contains k of the estimated
        eigenvectors from the active subspace analysis.
    :param ndarray W_br: m-by-(2*k) matrix that contains estimated upper and
        lower bounds on the components of the eigenvectors.
    :param str[] in_labels: A list of labels for the simulation's inputs.
    :param str out_label: A label for the quantity of interest.
    :param dict opts: A dictionary with some plot options.

    **See Also**

    utils.plotters.subspace_errors
    utils.plotters.eigenvalues

    **Notes**

    This function will plot at most the first four eigevectors in a four-subplot
    figure. In other words, it only looks at the first four columns of `W`.
    """
    if opts == None:
        opts = plot_opts()


    m, n = W.shape

    if W_br is not None:
        m_br, n_br = W_br.shape
        if n_br != 2*n:
            raise Exception('Number bootstrap range columns is wrong. Should be n_br=2n')

    # set labels for plots
    if out_label is None:
        out_label = 'Output'

    if n==1:
        plt.figure(figsize=(7,7))
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,0], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,0], W_br[:,1],
                facecolor='0.7', interpolate=True)
        plt.ylabel('Eigenvector 1 components')
        plt.title(out_label)
        plt.grid(True)
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

    elif n==2:
        plt.figure(figsize=(7,7))
        plt.subplot(211)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,0], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,0], W_br[:,1],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 1')
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            labelbottom='off') # labels along the bottom edge are off
        plt.axis([1, m, -1, 1])

        plt.subplot(212)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,1], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,2], W_br[:,3],
                facecolor='0.7', interpolate=True)
        plt.grid(True)
        plt.title(out_label + ', evec 2')
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

    elif n==3:
        plt.figure(figsize=(7,7))
        plt.subplot(221)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,0], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,0], W_br[:,1],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 1')
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            labelbottom='off') # labels along the bottom edge are off
        plt.axis([1, m, -1, 1])

        plt.subplot(222)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,1], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,2], W_br[:,3],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 2')
        plt.grid(True)
        plt.tick_params(axis='y', labelleft='off')
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

        plt.subplot(223)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,2], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,4], W_br[:,5],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 3')
        plt.grid(True)
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

    else:
        plt.figure(figsize=(7,7))
        plt.subplot(221)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,0], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,0], W_br[:,1],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 1')
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            labelbottom='off') # labels along the bottom edge are off
        plt.axis([1, m, -1, 1])

        plt.subplot(222)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,1], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,2], W_br[:,3],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 2')
        plt.grid(True)
        plt.tick_params(labelleft='off', labelbottom='off')
        plt.axis([1, m, -1, 1])

        plt.subplot(223)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,2], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,4], W_br[:,5],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 3')
        plt.grid(True)
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

        plt.subplot(224)
        plt.rc('font', **opts['myfont'])
        plt.plot(range(1, m+1), W[:,3], 'ko-', markersize=12)
        if W_br is not None:
            plt.fill_between(range(1, m+1), W_br[:,6], W_br[:,7],
                facecolor='0.7', interpolate=True)
        plt.title(out_label + ', evec 4')
        plt.grid(True)
        plt.tick_params(axis='y', labelleft='off')
        if in_labels is not None:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])

    if opts['savefigs']:
        figname = 'figs/evecs_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    show_plot(plt)

def sufficient_summary(y, f, out_label=None, opts=None):
    """
    Make a summary plot with the given predictors and responses.

    :param ndarray y: M-by-1 or M-by-2 matrix that contains the values of the
        predictors for the summary plot.
    :param ndarray f: M-by-1 matrix that contains the corresponding responses.
    :param str out_label: A label for the quantity of interest.
    :param dict opts: A dictionary with some plot options.

    **Notes**

    If `y.shape[1]` is 1, then this function produces only the univariate
    summary plot. If `y.shape[1]` is 2, then this function produces both the
    univariate and the bivariate summary plot, where the latter is a scatter
    plot with the first column of `y` on the horizontal axis, the second
    column of `y` on the vertical axis, and the color corresponding to `f`.
    """
    if opts == None:
        opts = plot_opts()

    # check sizes of y
    n = y.shape[1]
    if n == 1:
        y1 = y
    elif n == 2:
        y1 = y[:,0]
        y2 = y[:,1]
    else:
        raise Exception('Sufficient summary plots cannot be made in more than 2 dimensions.')

    # set labels for plots
    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=(7,7))
    plt.rc('font', **opts['myfont'])
    plt.plot(y1, f, 'bo', markersize=12)
    plt.xlabel('Active variable')
    plt.ylabel(out_label)
    plt.grid(True)
    if opts['savefigs']:
        figname = 'figs/ssp1_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if n==2:

        plt.figure(figsize=(7,7))
        plt.rc('font', **opts['myfont'])
        plt.scatter(y1, y2, c=f, s=150.0, vmin=np.min(f), vmax=np.max(f))
        plt.xlabel('Active variable 1')
        plt.ylabel('Active variable 2')
        ymin = 1.1*np.amin([np.amin(y1), np.amin(y2)])
        ymax = 1.1*np.amax([np.amax(y1), np.amax(y2)])
        plt.axis([ymin, ymax, ymin, ymax])
        plt.axes().set_aspect('equal')
        plt.grid(True)
        plt.title(out_label)
        plt.colorbar()
        if opts['savefigs']:
            figname = 'figs/ssp2_' + out_label + opts['figtype']
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)


    show_plot(plt)
def zonotope_2d_plot(vertices, design=None, y=None, f=None, out_label=None, opts=None):
    """
    A utility for plotting (m,2) zonotopes with associated designs and
    quadrature rules.

    :param ndarray vertices: M-by-2 matrix that contains the vertices that
        define the zonotope.
    :param ndarray design: N-by-2 matrix that contains a design-of-experiments
        on the zonotope. The plot will contain the Delaunay triangulation of the
        points in `design` and `vertices`.
    :param ndarray y: K-by-2 matrix that contains points to be plotted inside
        the zonotope. If `y` is given, then `f` must be given, too.
    :param ndarray f: K-by-1 matrix that contains a color value for the
        associated points in `y`. This is useful for plotting function values
        or quadrature rules with the zonotope. If `f` is given, then `y` must
        be given, too.
    :param str out_label: A label for the quantity of interest.
    :param dict opts: A dictionary with some plot options.

    **Notes**

    This function makes use of the scipy.spatial routines for plotting the
    zonotopes.
    """
    if opts == None:
        opts = plot_opts()

    # set labels for plots
    if out_label is None:
        out_label = 'Output'

    if vertices.shape[1] != 2:
        raise Exception('Zonotope vertices should be 2d.')

    if design is not None:
        if design.shape[1] != 2:
            raise Exception('Zonotope design should be 2d.')

    if y is not None:
        if y.shape[1] != 2:
            raise Exception('Zonotope design should be 2d.')

    if (y is not None and f is None) or (y is None and f is not None):
        raise Exception('You need both y and f to plot.')

    if y is not None and f is not None:
        if y.shape[0] != f.shape[0]:
            raise Exception('Lengths of y and f are not the same.')

    # get the xlim and ylim
    xmin, xmax = np.amin(vertices), np.amax(vertices)

    # make the Polygon patch for the zonotope
    ch = ConvexHull(vertices)

    # make the Delaunay triangulation
    if design is not None:
        points = np.vstack((design, vertices))
        dtri = Delaunay(points)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    fig0 = convex_hull_plot_2d(ch, ax=ax)
    for l in fig0.axes[0].get_children():
        if type(l) is Line2D:
	        l.set_linewidth(3)

    if design is not None:
        fig1 = delaunay_plot_2d(dtri, ax=ax)
        for l in fig1.axes[0].get_children():
            if type(l) is Line2D:
	        l.set_color('0.75')

    if y is not None:
        plt.scatter(y[:,0], y[:,1], c=f, s=100.0, vmin=np.min(f), vmax=np.max(f))
        plt.axes().set_aspect('equal')
        plt.title(out_label)
        plt.colorbar()

    plt.axis([1.1*xmin,1.1*xmax,1.1*xmin,1.1*xmax])
    plt.xlabel('Active variable 1')
    plt.ylabel('Active variable 2')
    show_plot(plt)
    if opts['savefigs']:
        figname = 'figs/zonotope_2d_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

def show_plot(plot, opts=None):
    plot.show()

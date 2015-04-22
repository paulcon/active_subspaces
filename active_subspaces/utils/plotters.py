import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, Delaunay, convex_hull_plot_2d, delaunay_plot_2d
import os

def plot_opts(savefigs=True, figtype='.eps'):
    # make figs directory
    if savefigs:
        if not os.path.isdir('figs'):
            os.mkdir('figs')

    # set plot fonts
    myfont = {'family' : 'arial',
            'weight' : 'normal',
            'size' : 14}
    
    return {'figtype' : figtype,
            'savefigs' : savefigs,
            'myfont' : myfont}
    
def eigenvalues(e, e_br=None, out_label=None, opts=None):

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
    plt.show()

def subspace_errors(sub_br ,out_label=None, opts=None):
    """
    subspace_errors plots the bootstrap errors in the subspace estimation

    Arguments:
        sub_br: An array with three columns. The first is the lower
                bootstrap bound. The second is the mean of the bootstrap
                error. The third is the upper bootstrap bound. 
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
    plt.axis([1, kk, np.amin(sub_br[:,0]), 1])
    
    if opts['savefigs']:
        figname = 'figs/subspace_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

def eigenvectors(W, W_br=None, in_labels=None, out_label=None, opts=None):

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
    plt.show()

def sufficient_summary(y, f, out_label=None, opts=None):

    if opts == None:
        opts = plot_opts()

    # check sizes of y
    n = y.shape[1]
    if n == 1:
        y1 = y
    else:
        y1 = y[:,0]
        y2 = y[:,1]

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
        plt.title(out_label)
        plt.colorbar()
        if opts['savefigs']:
            figname = 'figs/ssp2_' + out_label + opts['figtype']
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()
    
def zonotope_2d_plot(vertices, design=None, y=None, f=None, out_label=None, opts=None):
    
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
    convex_hull_plot_2d(ch, ax=ax)
    
    if design is not None:
        fig = delaunay_plot_2d(dtri, ax=ax)
        
    if y is not None:
        plt.scatter(y[:,0], y[:,1], c=f, s=100.0, vmin=np.min(f), vmax=np.max(f))
        plt.axes().set_aspect('equal')
        plt.title(out_label)
        plt.colorbar()
        
    plt.axis([1.1*xmin,1.1*xmax,1.1*xmin,1.1*xmax])
    plt.xlabel('Active variable 1')
    plt.ylabel('Active variable 2')
    plt.show()
    if opts['savefigs']:
        figname = 'figs/zonotope_2d_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    
    
        
    
    
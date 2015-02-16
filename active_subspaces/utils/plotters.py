import numpy as np
import matplotlib.pyplot as plt
import os

def plot_opts(savefigs=True, figtype='eps'):
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

    plt.figure()
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1 ,k+1), e, 'ko-')
    if e_br is not None:
        plt.fill_between(range(1, k+1), e_br[:,0], e_br[:,1],
            facecolor='0.7', interpolate=True)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalues')
    plt.title(out_label)
    plt.grid(True)
    plt.xticks(range(1, k+1))
    if e_br is None:
        plt.axis([1, k, np.amin(e), np.amax(e)])
    else:
        plt.axis([1, k, np.amin(e_br[:,0]), np.amax(e_br[:,1])])
    figname = 'figs/evals_' + out_label + opts['figtype']
    if opts['savefigs']:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

def subspace_errors(sub_br ,out_label=None, opts=None):

    if opts == None:
        opts = plot_opts()

    kk = sub_br.shape[0]
    if out_label is None:
        out_label = 'Output'

    plt.figure()
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1, kk+1), sub_br[:,1], 'ko-', markersize=12)
    plt.fill_between(range(1, kk+1), sub_br[:,0], sub_br[:,2],
        facecolor='0.7', interpolate=True)
    plt.xlabel('Subspace dimension')
    plt.ylabel('Subspace distance')
    plt.grid(True)
    plt.xticks(range(1, kk+1))
    plt.axis([1, kk, np.amin(sub_br[:,0]), 1])
    figname = 'figs/subspace_' + out_label + opts['figtype']
    if opts['savefigs']:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

def eigenvectors(W, W_boot=None, in_labels=None, out_label=None, opts=None):

    if opts == None:
        opts = plot_opts()

    n = len(W.shape)
    m = W.shape[0]
    # set labels for plots
    if in_labels is None:
        in_labels = [str(i) for i in range(1,m+1)]
    if out_label is None:
        out_label = 'Output'

    if n==1:
        plt.figure()
        plt.rc('font', **opts['myfont'])
        if W_boot is not None:
            plt.plot(range(1 ,m+1), W_boot, color='0.7')
        plt.plot(range(1, m+1), W, 'ko-', markersize=12)
        plt.ylabel('Weights')
        plt.grid(True)
        if m<=10:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])
        figname = 'figs/evecs_' + out_label + opts['figtype']
        if opts['savefigs']:
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    else:
        plt.figure()
        plt.rc('font', **opts['myfont'])
        for k in range(np.minimum(3, W.shape[1])):
            plt.plot(range(1, m+1), W[:,k], 'o-', markersize=12, label='%d' % k)
        plt.ylabel('Eigenvectors')
        plt.grid(True)
        if m<=10:
            plt.xticks(range(1, m+1), in_labels, rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1, m, -1, 1])
        plt.legend(loc='best')
        figname = 'figs/evecs_' + out_label + opts['figtype']
        if opts['savefigs']:
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()

def sufficient_summary(y, f, out_label=None, opts=None):

    if opts == None:
        opts = plot_opts()

    # check sizes of y
    n = len(y.shape)
    if n == 1:
        y1 = y
    else:
        y1 = y[:,0]
        y2 = y[:,1]

    # set labels for plots
    if out_label is None:
        out_label = 'Output'

    plt.figure()
    plt.rc('font', **opts['myfont'])
    plt.plot(y1, f, 'bo', markersize=12)
    plt.xlabel('Active variable')
    plt.ylabel(out_label)
    plt.grid(True)
    figname = 'figs/ssp1_' + out_label + opts['figtype']
    if opts['savefigs']:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if n==2:

        plt.figure()
        plt.rc('font', **opts['myfont'])
        plt.scatter(y1, y2, c=f, s=150.0, vmin=np.min(f), vmax=np.max(f))
        plt.xlabel('Active variable 1')
        plt.ylabel('Active variable 2')
        plt.title(out_label)
        plt.colorbar()
        figname = 'figs/ssp2_' + out_label + opts['figtype']
        if opts['savefigs']:
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()
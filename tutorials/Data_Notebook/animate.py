import numpy as np
import matplotlib.pyplot as plt
import active_subspaces as asub
import matplotlib.animation as anim

def animate(XX, F, w, br, dpi, name='animation.mp4', in_labels=None, out_label=None):

    M,m = XX.shape
    
    f = F
    
    frames_per_second = 30
    seconds_per_variable = 2
    frames_per_segment = frames_per_second*seconds_per_variable
    
    fig,axes = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)
    axes[0].axis([-2, 2, .9*np.amin(f), 1.1*np.amax(f)])
    axes[1].axis([0, m+1, -1.1, 1.1])
    axes[1].set_xticks(np.arange(1, m+2))
    axes[1].set_xticklabels(in_labels, rotation='vertical')
    axes[1].margins(0.2)
    fig.subplots_adjust(bottom=0.2)
    axes[0].grid()
    axes[1].grid()
    axes[0].set_xlabel(r'Active Variable')
    axes[0].set_ylabel(out_label)
    axes[1].set_ylabel('Weights of active variable')
    line0, = axes[0].plot([], [], 'bo', markersize=8)
    line1, = axes[1].plot(np.arange(1, m+1), np.squeeze(w), 'ko', markersize=12, markeredgecolor='k', markeredgewidth=2, markerfacecolor='none')
    axes[1].legend(['Active Subspace'] ,loc=3)
    line2, = axes[1].plot([], [], 'ro', markersize=10)
    def ifunc():
        line0.set_data([], [])
        line2.set_data([], [])
        return line0, line2
        
    def afunc(i):
        if i < frames_per_segment*(m-1):
            x2 = np.arange(1, m+1)
            y2 = np.zeros_like(x2, dtype='float64')
            dvarind = i/frames_per_segment
            ivarind = dvarind + 1
            y2[ivarind] = i%frames_per_segment/(1.0*frames_per_segment)
            y2[dvarind] = np.sqrt(1 - y2[ivarind]**2)
            line2.set_data(x2, y2)
            x0 = np.dot(y2, XX.T)
            line0.set_data(x0, f)
        elif i < frames_per_segment*m:
            x2 = np.arange(1, m+1)
            y2 = np.zeros_like(x2, dtype='float64')
            y2[m-1] = 1
            y2f = np.squeeze(w)
            ydiff = y2 - y2f
            y2 = y2 - i%frames_per_segment/(frames_per_segment*1.0)*ydiff
            line2.set_data(x2, y2)
            x0 = np.dot(y2, XX.T)
            line0.set_data(x0, f)
        else:
            x2 = np.arange(1, m+1)
            y2 = np.zeros_like(x2,dtype='float64')
            y2f = np.squeeze(w)
            y2 = y2f
            line2.set_data(x2, y2)
            x0 = np.dot(y2, XX.T)
            line0.set_data(x0, f)
        return line0, line2
        
    movie = anim.FuncAnimation(fig, afunc, frames=frames_per_segment*(m+1), init_func=ifunc, repeat=False, blit=False, interval=33.333)
    movie.save(name, bitrate=br, dpi=dpi)
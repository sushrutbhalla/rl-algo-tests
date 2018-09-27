import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3, add_pad=False):
    '''Compute the moving average of array a using a window of size n
    add_pad: if True, a padding to the final array is added to make it equal in size to input array
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    final = ret[n - 1:] / n
    if not add_pad:
        return final
    else:
        pad = final[-1]*np.ones(len(a)-len(final))
        return np.concatenate((final,pad),axis=0)

def plot_avg_cumulative_reward(avg_cumulative_reward, legend, title, filename=None, smooth=False, n=10, use_ax_limit=True, ymin=-20, ymax=220):
    ''' Plot the list of list of avg_cumulative_reward on a single graph.
    Most of the input variable names are fairly intuitive and others are detailed below:
    filename: specify a filename if the plot should be saved
    smooth: generate a smooth graph using moving average of n steps
    n: number of steps to use to generate a moving average
    use_ax_limit: limit the x and y axes values with a maximum and minimum (ymin and ymax)
    '''
    for idx in range(len(avg_cumulative_reward)):
        if not smooth:
            plt.plot(avg_cumulative_reward[idx])
        else:
            plt.plot(moving_average(avg_cumulative_reward[idx],n=n))
    plt.title(title)
    if not smooth:
        plt.ylabel('Cumulative Reward')
    else:
        plt.ylabel('Smoothed({}) Cumulative Reward'.format(n))
    plt.xlabel('Episode')
    if use_ax_limit:
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
    plt.legend(legend, loc='lower right')
    if filename is not None:
        plt.savefig(filename)
    plt.show()




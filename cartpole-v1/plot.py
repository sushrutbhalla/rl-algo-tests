from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import sys

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


def plot_cumulative_reward(avg_cumulative_reward, legend, title, filename, avg_rew=True, smooth=False, n=10, use_ax_limit=True, ymin=-20, ymax=220):
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
    y_plt_label = ''
    if not smooth:
        y_plt_label = 'Cumulative Reward'
    else:
        y_plt_label = 'Smoothed({}) Cumulative Reward'.format(n)
    if avg_rew:
        y_plt_label = 'Avg: ' + y_plt_label
    plt.ylabel(y_plt_label)
    plt.xlabel('Episode')
    if use_ax_limit:
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
    plt.legend(legend, loc='lower right')
    # plt.legend(legend, loc='upper right')
    if filename is not None:
        plt.savefig(filename)
    plt.show()

#read result file
assert len(sys.argv) >=3, "missing filename"
file_name = sys.argv[1]
file_name2 = sys.argv[2]

records = np.genfromtxt(file_name, delimiter=',')
cumulative_rewards = records[:1500]
records2 = np.genfromtxt(file_name2, delimiter=',')
cumulative_rewards2 = records2[:1500]

plot_legend = []
plot_legend.append('DQN on Cartpole-v1')
plot_legend.append('DQN on Cartpole-v1 w/ Gradient Clip')

############################################################
#plot results

plot_filename = 'dqn_cartpole_comp.png'
plot_title = "DQN: Original Cumulative Reward"
plot_cumulative_reward([cumulative_rewards, cumulative_rewards2],
  plot_legend, plot_title, plot_filename, avg_rew=False, use_ax_limit=False)

#plot smoothed curves
smooth_low=10
smooth_high=40
plot_filename = 'dqn_cartpole_smooth_10_comp.png'
plot_title = "DQN: Smooth(n={}) Cumulative Reward".format(smooth_low)
plot_cumulative_reward([cumulative_rewards, cumulative_rewards2],
  plot_legend, plot_title, plot_filename, n=smooth_low, avg_rew=False, smooth=True, use_ax_limit=False)

plot_filename = 'dqn_cartpole_smooth_40_comp.png'
plot_title = "DQN: Smooth(n={}) Cumulative Reward".format(smooth_high)
plot_cumulative_reward([cumulative_rewards, cumulative_rewards2],
  plot_legend, plot_title, plot_filename, n=smooth_high, avg_rew=False, smooth=True, use_ax_limit=False)

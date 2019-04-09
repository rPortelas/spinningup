import matplotlib.patches as patches
from param_env_utils.imgep_utils.gep_utils import scale_vector
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import imageio
import numpy as np
import copy
import os

def scatter_plot(data, ax=None, emph_data=None, xlabel='min stump height', ylabel='max stump height', xlim=None, ylim=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(7, 7))
    Xs, Ys = [d[0] for d in data], [d[1] for d in data]
    if emph_data is not None:
        emphXs, emphYs = [d[0] for d in emph_data], [d[1] for d in emph_data]
    ax.plot(Xs, Ys, 'r.', markersize=2)
    ax.axis('equal')
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if emph_data is not None:
        ax.plot(emphXs, emphYs, 'b.', markersize=5)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

def plot_regions(boxes, interests, ax=None, xlabel='min stump height', ylabel='max stump height', xlim=None, ylim=None):
    # Create figure and axes
    if ax == None:
        f, ax = plt.subplots(1, 1, figsize=(8, 7))
    # Add the patch to the Axes
    for b, ints in zip(boxes, interests):
        # print(b)
        lx, ly = b.low
        hx, hy = b.high
        c = plt.cm.jet(ints)
        rect = patches.Rectangle([lx, ly], (hx - lx), (hy - ly), linewidth=3, edgecolor='white', facecolor=c)
        ax.add_patch(rect)
        # plt.Rectangle([lx,ly],(hx - lx), (hy - ly))

    cax, _ = cbar.make_axes(ax)
    cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet)
    cb.set_label('Mean Competence Progress')
    ax.axis('equal')
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)


def region_plot_gif(all_boxes, interests, iterations, goals,
                    gifname='saggriac', rewards=None, ep_len=None, gifdir='graphics/', xlim=[0,1], ylim=[0,1]):
    plt.ioff()
    print("Making an exploration GIF: " + gifname)
    # Create target Directory if don't exist
    tmpdir = 'tmp/'
    tmppath = gifdir + 'tmp/'
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
        print("Directory ", tmppath, " Created ")
    else:
        print("Directory ", tmppath, " already exists")
    filenames = []
    images = []
    steps = []
    mean_rewards = []
    plot_step = 250
    for i in range(len(goals)):
        if i > 0 and (i % plot_step == 0):
            f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(27, 7))
            ax = [ax0, ax1, ax2]
            scatter_plot(goals[0:i], ax=ax[0], emph_data=goals[i - plot_step:i], xlim=xlim, ylim=ylim)
            idx = 0
            cur_idx = 0
            for j in range(len(all_boxes)):
                if iterations[j] > i:
                    break
                else:
                    cur_idx = j

            # ADD TRAINING CURVE
            ax[2].set_ylabel('Train return', fontsize=18)
            steps.append(sum(ep_len[0:i]))
            mean_rewards.append(np.mean(rewards[i - plot_step:i]))
            ax[2].plot(steps, mean_rewards)

            plot_regions(all_boxes[cur_idx], interests[cur_idx], ax=ax[1], xlim=xlim, ylim=ylim)

            f_name = gifdir+tmpdir+"scatter_{}.png".format(i)
            plt.suptitle('Episode {}'.format(i), fontsize=20)
            plt.savefig(f_name, bbox_inches='tight')
            plt.close(f)
            filenames.append(f_name)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.3)
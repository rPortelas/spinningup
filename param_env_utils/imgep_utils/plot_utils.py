import matplotlib.patches as patches
from param_env_utils.imgep_utils.gep_utils import scale_vector
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import imageio
import numpy as np
import copy
import os
from matplotlib.patches import Ellipse

def plt_2_rgb(ax):
    ax.figure.canvas.draw()
    data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
    return data


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
    print(boxes)
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
            print(i)
            for j in range(len(all_boxes)):
                if iterations[j] > i:
                    break
                else:
                    cur_idx = j
            print(j)
            # ADD TRAINING CURVE
            ax[2].set_ylabel('Train return', fontsize=18)
            steps.append(sum(ep_len[0:i]))
            mean_rewards.append(np.mean(rewards[i - plot_step:i]))
            ax[2].plot(steps, mean_rewards)

            plot_regions(all_boxes[cur_idx], interests[cur_idx], ax=ax[1], xlim=xlim, ylim=ylim)

            f_name = gifdir+tmpdir+"scatter_{}.png".format(i)
            plt.suptitle('Episode {}'.format(i), fontsize=20)
            images.append(plt_2_rgb(plt.gca()))
            plt.close(f)
    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.3)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    covariance = covariance[0:2,0:2]
    position = position[0:2]

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(2, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def draw_competence_grid(ax, comp_grid, x_bnds, y_bnds):
    ax.pcolor(x_bnds, y_bnds, np.transpose(comp_grid),cmap=plt.cm.gray, edgecolors='k', linewidths=2,
              alpha=0.3)
    cax, _ = cbar.make_axes(ax,location='left')
    cb = cbar.ColorbarBase(cax, cmap=plt.cm.gray)
    cb.set_label('Competence')
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

def plot_gmm(weights, means, covariances, X, ax=None, xlim=[0,1], ylim=[0,1], xlabel='jkl', ylabel='jhgj', bar=True):
    ax = ax or plt.gca()
    #colors = [plt.cm.jet(i) for i in X[:, -1]]
    colors = [plt.cm.autumn_r(i) for i in X[:, -1]]
    sizes = [5+np.interp(i,[0,1],[0,10]) for i in X[:, -1]]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=sizes, zorder=2)
    #ax.axis('equal')
    w_factor = 0.6 / weights.max()
    for pos, covar, w in zip(means, covariances, weights):
        draw_ellipse(pos, covar, alpha=0.6, ax=ax)

    #plt.margins(0, 0)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if bar:
        cax, _ = cbar.make_axes(ax)
        cb = cbar.ColorbarBase(cax, cmap=plt.cm.autumn_r)
        cb.set_label('Learning progress', fontsize=25)
        cax.tick_params(labelsize=20)

    ax.set_xlabel('stump height', fontsize=25)
    #ax.set_ylabel('spacing', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

def gmm_plot_gif(bk, gifname='test', gifdir='graphics/', ax=None,
                 xlim=[0,1], ylim=[0,1], fig_size=(9,6), save_imgs=False, title=True, bar=True):
    plt.ioff()
    # Create target Directory if don't exist
    tmpdir = 'tmp/'
    tmppath = gifdir + 'tmp/'
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
        print("Directory ", tmppath, " Created ")
    else:
        print("Directory ", tmppath, " already exists")
    print("Making " + tmppath + gifname + ".gif")
    images = []
    old_ep = 0
    gen_size = int(len(bk['goals_lps']) / len(bk['episodes']))
    gs_lps = bk['goals_lps']
    for i,(ws, covs, means, ep) in enumerate(zip(bk['weights'], bk['covariances'], bk['means'], bk['episodes'])):
            plt.figure(figsize=fig_size)
            ax = plt.gca()
            plot_gmm(ws, means, covs, np.array(gs_lps[old_ep+gen_size:ep+gen_size]),
                     ax=ax, xlim=xlim, ylim=ylim, bar=bar)  #add gen_size to have gmm + the points that they generated, not they fitted
            if 'comp_grid' in bk:  # add competence grid info
                draw_competence_grid(ax,bk['comp_grids'][i], bk['comp_xs'][i], bk['comp_ys'][i])
            f_name = gifdir+tmpdir+gifname+"_{}.png".format(ep)
            if title:
                plt.suptitle('Episode {} | nb gaussians:{}'.format(ep,len(means)), fontsize=20)
            old_ep = ep
            if save_imgs: plt.savefig(f_name, bbox_inches='tight')
            images.append(plt_2_rgb(ax))
            plt.close()

    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.3)


def plot_cmaes(mean, covariance, ints, X, sigma, currX=None, currInts=None,
               ax=None, xlim=[0.,1.], ylim=[0,1], xlabel='jkl', ylabel='jhgj'):
    ax = ax or plt.gca()
    if len(ints) > 0:
        colors = [plt.cm.jet(i) for i in ints]
        #ax.scatter(X[:, 0], X[:, 1],c=colors, s=1, zorder=2, alpha=0.5)
    #ax.axis('equal')
    if currX is not None:
        currColors = [plt.cm.jet(i) for i in currInts]
        ax.scatter(currX[:, 0], currX[:, 1],c=currColors, s=5, zorder=2)
    draw_ellipse(mean, (sigma**2) * covariance, alpha=0.5)

    cax, _ = cbar.make_axes(ax)
    cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet)
    cb.set_label('Interest')
    ax.axis('equal')
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

def cmaes_plot_gif(bk, gifname='testcmaes', gifdir='graphics/', ax=None):
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
    bk['goals'] = np.array(bk['goals'])
    prev_ep = 0
    for cov, mean, ep, sig in zip(bk['covariances'], bk['means'], bk['episodes'], bk['sigmas']):
            ax = plt.gca()
            plot_cmaes(mean, cov, bk['interests'][0:ep], bk['goals'][0:ep], sig,
                       currX=bk['goals'][prev_ep:ep],
                       currInts=bk['interests'][prev_ep:ep], ax=ax)
            prev_ep = ep
            f_name = gifdir+tmpdir+"scatter_{}.png".format(ep)
            plt.suptitle('Episode {}'.format(ep), fontsize=20)
            images.append(plt_2_rgb(ax))
            plt.close()
    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.3)

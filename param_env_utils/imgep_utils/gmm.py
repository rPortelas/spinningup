import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from sklearn.datasets import make_moons
import numpy as np
import copy
from gym.spaces import Box
from param_env_utils.imgep_utils.dataset import BufferedDataset
from param_env_utils.imgep_utils.gep_utils import proportional_choice

class EmpiricalLearningProgress():
    def __init__(self, goal_size):
        self.interest_knn = BufferedDataset(1, goal_size, buffer_size=500, lateness=0)

    def get_lp(self, goal, competence):
        interest = 0
        if len(self.interest_knn) > 5:
            # compute learning progress for new goal
            dist, idx = self.interest_knn.nn_y(goal)
            # closest_previous_goal = previous_goals[idx]
            closest_previous_goal = self.interest_knn.get_y(idx[0])
            closest_previous_goal_competence = self.interest_knn.get_x(idx[0])
            # print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)

            # compute Progress as absolute difference in competence
            progress = closest_previous_goal_competence - competence
            interest = np.abs(progress)

        # add to database
        self.interest_knn.add_xy(competence, goal)
        return interest

class InterestGMM():
    def __init__(self, min, max, n_components, seed=None):
        if not seed:
            seed = np.random.randint(424242)
        self.gmm = GMM(n_components=n_components, covariance_type='full', random_state=seed)
        self.random_goal_generator = Box(np.array(min), np.array(max), dtype=np.float32)
        self.lp_computer = EmpiricalLearningProgress(len(min))
        self.goals = []
        self.lps = []
        self.goals_lps = []


    def compute_interests(self, sub_regions):
        pass

    def update(self, goals, competences):
        # if not isinstance(competences, list):
        #     competences = [competences]
        # if not isinstance(goals[0], list):
        #     goals = [goals]
        for g, c in zip(goals, competences):
            self.goals.append(g)
            self.lps.append(self.lp_computer.get_lp(g, c))
            self.goals_lps.append(g+self.lps[-1])

        #re-fit
        self.gmm.fit(self.goals)

    def sample_goal(self, n_samples=1):
        self.lp_means = []
        self.lp_stds = []
        for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            self.lp_means.append(pos[-1])
            self.lp_stds.append(covar[-1,-1])
        goals = []
        for _ in range(n_samples):
            # sample gaussian
            idx = proportional_choice(self.lp_means, eps=0.2)
            # sample goal in gaussian
            goals.append(np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx]))
        return np.array(goals)

    def old_sample_goal(self, n_samples=1):
        Xnew, _ = self.gmm.sample(n_samples=n_samples)
        return Xnew


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


# Generate some data
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)

gmm16 = InterestGMM([0,0],[1,1], n_components=16)
#plot_gmm(gmm16, Xmoon, label=False)
#plt.show(block=False)
#plt.pause(1.0)
gmm16.update(Xmoon, [0.]*len(Xmoon))
Xnew = gmm16.sample_goal(n_samples=400)
plt.scatter(Xnew[:, 0], Xnew[:, 1])
plt.show(block=False)
plt.pause(2.0)
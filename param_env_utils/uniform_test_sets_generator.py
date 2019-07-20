import numpy as np
import pickle
from param_env_utils.test_utils import get_test_set_name
import copy

np.random.seed(4222)
sample_size = 50
arg_ranges = {'roughness':None,
              'stump_height':None,#[0.3,0.66],#None,#[0,4.0],#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
              'stump_width':None,#[3.0,5.0],
              'tunnel_height':None,
              'obstacle_spacing':None,#[0,6.0],
              'poly_shape':None,#[0,4.0],#[0,10.0],
              'gap_width':None,
              'step_height':None,
              'step_number':None,
              'stump_seq':[0,6.0]}

name = get_test_set_name(arg_ranges)
test_env_list = []
# --- sample environments uniformly
points = []
for i in range(sample_size):
    test_env = arg_ranges.copy()
    point = []
    for k,v in test_env.items():
        if v is not None:
            if k == "poly_shape":
                point.append(np.random.uniform(v[0],v[1],12))
            elif k == 'stump_seq':
                point.append(np.random.uniform(v[0], v[1], 10))
            else:
                point.append(np.random.uniform(v[0], v[1]))
            test_env[k] = point[-1]
    points.append(point)
    test_env_list.append(test_env)

# if len(points[0]) <= 2:
#     import matplotlib.pyplot as plt
#     plt.plot([p[0] for p in points], [p[1] for p in points], 'o')
#     plt.axis("equal")
#     plt.show()

print(test_env_list)
pickle.dump(test_env_list, open('test_sets/'+name+".pkl", "wb"))
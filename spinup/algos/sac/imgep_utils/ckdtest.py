import time
import numpy as np
import scipy.spatial
import pickle

# for i in range(500):
#     start = time.time()
#     knn = scipy.spatial.cKDTree(np.random.random((10000,31)))
#     end = time.time()
#     print(end-start)

favorite_color = pickle.load( open( "../save_weird_array.p", "rb" ) )
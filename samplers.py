import numpy as np
from scipy.spatial.distance import cdist

def naive_sampler(device, max_num_points, x_0, pos_dir=False):
    x_p=[]
    x_m=[]
    vs = []
    while(len(x_p) < max_num_points):
        p = np.random.randn(x_0.shape[0])
        if pos_dir:
            p = np.exp(2*p)
        x_mi,x_pi, get_v = device.line_search(x_0, p)
        x_m.append(x_mi)
        x_p.append(x_pi)
        vs.append(get_v)
    x_p = np.array(x_p)
    x_m = np.array(x_m)
    return x_m, x_p, vs
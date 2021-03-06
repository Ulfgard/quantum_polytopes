from autograd import numpy as np
from autograd import grad
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from samplers import naive_sampler
import cvxpy as cp

# Numerically more stable way to compute log of sum of exponentials
def logsumexp(xs):
    xmax = np.max(xs, axis = 1)
    expxs = np.exp(xs - xmax.reshape(-1,1))
    return np.log(np.sum(expxs,axis=1)) + xmax
    
def solve_max_likelihood_problem(x_m, x_p, A_init, b_init, delta):
    d = A_init.shape[1]
    m = A_init.shape[0]
    
    def log1pexp(x):
        return np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0)

    def neg_log_loss(A, b, x, y):
        #linear activations
        act = x[:,:A.shape[1]]@A.T + b.reshape(1,-1)
        
        #log-nonlinear activations
        act = y * logsumexp(act)
        #loss
        return log1pexp(-act)
        
    def obj(x):
        A=x[:d*m].reshape(m,d)
        q=x[d*m:]

        #compute log likelihood
        nll = 0.0
        if x_m.shape[0] > 0:
            b = -A@q
            nll_m = neg_log_loss(A,b,x_m,-1)
            nll_p = neg_log_loss(A,b,x_p,1)
            
            #compute loss
            nll = np.sum(nll_m) + np.sum(nll_p)
            
        # Add penalty and return
        angular_penalty = np.sum((1-np.diag(A)/np.linalg.norm(A,axis=1))**2)
        position_penalty = np.sum(q**2)
        
        return nll + 100*angular_penalty + 100*position_penalty
    
    # Compute starting point and solve
    q_init = -np.linalg.inv(A_init) @ b_init
    x_init = np.zeros(d*m+m)
    x_init[:d*m] = A_init.reshape(-1)
    x_init[d*m:] = q_init.reshape(-1)
    x_res = minimize(obj, x_init, jac=grad(obj), tol=1.e-7).x
    
    A = x_res[:d*m].reshape(m,d)
    q = x_res[d*m:]
    return A, -A@q

def sample_model(device, A, b, origin):
    """
    Use the currently estimated model to decide in which directions 
    new line searches need to be performed, starting from the origin provided. 
    Then perform those, and return the newly obtained datapoints.
    """
    N = A.shape[0]
    d = A.shape[1]
    invA = np.linalg.inv(A)
    lenA = np.linalg.norm(A,axis=1)
    
    # Intersection of boundaries
    q = -invA@b # Fulfills Aq+b = 0
    
    # We sample directions along lines q+t*dir
    # The ith line fulfills that it lies on the ith plane,
    # and for all planes j != i it holds that 
    #     A_j^T(q+t*dir)+b <= 0
    # <=> A_j^T dir <= 0 (using Aq+b = 0)
    # We do this by sampling a rhs g with g[i] = 0 and g[j] <0, j != i and then dir = A^-1q
    # We choose g[j] in a way such that it is always "close" to some other boundary because
    # this might be close to vertex points/useful to check for wrong intersections of boundaries
    dirs = -np.exp(2*np.random.randn(N,d))
    dirs -= np.diag(np.diag(dirs))
    dirs *= lenA.reshape(-1,1)
    dirs = dirs @ invA.T
    dirs /= np.linalg.norm(dirs,axis=1).reshape(-1,1)
    
    # Find intersection of lines x= q+t*dirs[i] with the lines p_j == origin_j
    # Since q >= origin, we have t = min_j (origin_j - q_j)/dirs[i,j] over all j with dirs[i,j] <= 0
    ts = np.zeros(N)
    for i in range(N):
        sel = dirs[i] < 0
        ts[i] = np.min((origin-q)[sel]/dirs[i,sel])*np.random.uniform(0.0,1.0)
    
    # Compute intersections
    xs_model = q.reshape(1,-1) + np.diag(ts) @ dirs
    
    # Sample points
    x_m =[]
    x_p =[]
    for i in range(xs_model.shape[0]):
        p = xs_model[i,:] - origin
        
        x_mi, x_pi, v = device.line_search(origin,p)
        if not v is None:
            x_m.append(x_mi)
            x_p.append(x_pi)
            
    # Return as numpy arrays
    return np.array(x_p), np.array(x_m)

def naive_sampler(device, max_num_points, x_0, pos_dir=False):
    """
    Performs line searches in fully random directions, and records
    the hits. If pos_dir, only fire rays outwards.
    """
    x_p=[]
    x_m=[]
    vs = []
    
    while(len(x_p) < max_num_points):
        # Generate random direction
        p = np.random.randn(x_0.shape[0])
        if pos_dir:
            p = np.exp(2*p)
            
        # Do a line search, point just below and just above
        x_mi, x_pi, get_v = device.line_search(x_0, p)
        x_m.append(x_mi)
        x_p.append(x_pi)
        vs.append(get_v)
        
    return np.array(x_m), np.array(x_p), vs

def stopping_criterion(A, b, x_m, x_p, delta):
    """
    Evaluate whether the algorithm can terminate (meaning we have found
    the polytopes and facets to within some precision)
    """
    m = A.shape[0]
    d = A.shape[1]
    counts = np.zeros(m)
    norms = np.linalg.norm(A,axis=1)
    
    #count sample pairs that are separated by a plane
    f_m = (x_m@A.T + b.reshape(1,-1))/norms.reshape(1,-1)
    f_p = (x_p@A.T + b.reshape(1,-1))/norms.reshape(1,-1)
    separated = np.logical_and(f_m < 0.1*delta, f_p > -0.1*delta) #true if boundary passes between points
    
    for i in range(separated.shape[0]):
        #ignore points that are separated by multiple facets as they can be problematic
        if np.sum(separated[i]) == 1:
            counts[separated[i]] += 1
            
    # If there are d+1 close samples on all facets, we are done.
    # For peace of mind, we're not stopping until d+5 are close (and because this will lead to higher precision)
    stop = np.min(counts) >= d + 5
    return stop, counts, counts > (d + 5)
    
def learn_zero_polytope(device, delta, num_start_samples = None, max_searches = 4000):
    """
    The algorithm that learns the polytope of the '00...0' charge state of the device.
    It assumes that the device is in that charge state.
    """
    # Function to filter close duplicates
    def filter_close(x, y, delta):
        dist = cdist(x,x)
        keep = np.array([True]*x.shape[0])
        for i in range(x.shape[0]):
            if not keep[i]:
                continue
            for j in range(i+1, x.shape[0]):
                if dist[i,j] < delta:
                    keep[j] = False
        return x[keep==True,:],y[keep==True,:]
    
    lower_vertex = device.lower_bound
    d = device.num_inputs
    if num_start_samples is None:
        num_start_samples = d*d
    
    # Perform an initial set of fully random line searches to populate our dataset
    num_searches = num_start_samples
    x_m, x_p, _ = naive_sampler(device, num_start_samples, lower_vertex, True)
    
    # Initialize the solution as set of planes intersecting at the provided vertex
    iter_info=[]
    while True:
        # Fit a new polytope
        A = np.eye(d)/delta
        b = -np.max(A @ x_m.T, axis=1)
        A, b = solve_max_likelihood_problem(x_m, x_p, A, b, delta)

        # Use the model to perform new line searches for new datapoints
        x_p_new, x_m_new = sample_model(device, A, b, lower_vertex)
        
        # Add new points to dataset
        x_m = np.vstack([x_m,x_m_new])
        x_p = np.vstack([x_p,x_p_new])
        x_m, x_p = filter_close(x_m,x_p, 1.5*delta)
        
        # Keep track of how many line searches we performed
        num_searches += x_m_new.shape[0]
        
        # Check if we are done
        stop, counts, keep = stopping_criterion(A, b, x_m, x_p, delta)
        
        # Keep track of how many searches we did per iteration, and how many
        # points are close to the boundaries.
        iter_info.append([num_searches, counts])
        if stop or num_searches >= max_searches:
            return np.hstack([A,b.reshape(-1,1)]), x_m, x_p, iter_info
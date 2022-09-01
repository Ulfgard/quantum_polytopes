from autograd import numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp
from scipy.optimize import minimize
import cvxpy as cp
from scipy.spatial.distance import cdist
from samplers import naive_sampler

def sample_boundary_points(A_full, b_full, boundsA, boundsb, points_per_boundary=3, do_sample = None):
    d = A_full.shape[1]
    
    # Find points to sample. If None, we will sample all of the supplied facets
    if do_sample is None:
        do_sample = np.ones(A_full.shape[0])
    
    points = []
    mids = []
    rs = []
    for i in range(A_full.shape[0]):
        Ai = A_full[i].copy()
        bi = b_full[i]
        A = np.delete(A_full,i,axis=0)
        b = np.delete(b_full,i)

        # We have to sample a point on the facet of the constraint
        # To do this we compute the circle with maximum circumference inside the facet. For this we project 
        # the problem on the cut of the constraint with the remaining polytope. After we find the circle,
        # we can sample a random point in it and project it back into the original space
        
        # Step 2.1: Projection. We first rotate the coordinate constraint with a rotation Q so 
        # that Q@Ai (|Ai|,0,...0)=|Ai|*e_1
        # We then know that since points must fullfill Ai@x + bi =0, the rotated points fulfill
        # Ai^T @ Q^T @ (Q@x)= Ai^T@Q^T@y= |Ai|*e_1^Ty = |Ai| y_1
        # and thus the constraint becomes y_1 = -bi/|Ai| which allows to project the dimension away
        
        # Find Q
        # Q = I + [A_i; e_1 ] (R_theta - I)[A_i; e_1 ]^T, where R is a 2D rotation matrix with
        # theta=angle(A_i,e_i). See e.g.
        # https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
        
        V=np.zeros((2,d))
        V[0,:] = Ai / np.linalg.norm(Ai)
        V[1,0] = 1
        V[1,:] -= V[0,0]*V[0,:]
        
        # Check if we need to rotate at all
        if np.linalg.norm(V[1,:])< 1.e-5:
            Q = np.eye(d)
            Q[0,0] = np.sign(V[0,0])
        else:
            V[1,:] /= np.linalg.norm(V[1,:])

            cos_theta = V[0,0]
            sin_theta = (1-cos_theta**2)**0.5
            R=np.array([[cos_theta,-sin_theta],[sin_theta, cos_theta]])
            Q = np.eye(d) + V.T @ (R-np.eye(2)) @ V
        
        #rotate A
        A = A@Q.T
        
        # Find the circle with maximum circumference inside the projected facet
        # Since we project away the first dimension, norms need to exclude the first dimension.
        norm_vector = np.linalg.norm(A[:,1:], axis=1) 
        
        r = cp.Variable(1)
        y = cp.Variable(d)
        lin_constraints = [
            y[0] == -bi / np.linalg.norm(Ai),# We stay on the ith constraint
            A @ y + b + r*norm_vector <= 0,  # Linear boundaries are only allowed to intersect the sphere once
            (boundsA@Q.T)@y + r + boundsb <= 0, # Also take bound constraints into account
            r >= 0 # Radius needs to be positive
        ]
        
        #try to solve the problem
        prob = cp.Problem(cp.Maximize(r),constraints=lin_constraints)
        try:
            prob.solve(verbose=False,max_iters=10000)
            status = prob.status
        except:
            print("except:", prob.status)
            status = "infeasible"
        
        if status not in ["infeasible", "infeasible_inaccurate"]:
            r = r.value[0]
            y = y.value
        else: # SOLVING PROBLEM FAILED FOR SOME REASON.
        
            # Transition might not intersect. 
            # Find minimum offset to boundary to make it intersect at at least a single point
            y = cp.Variable(d)
            lin_constraints = [
                A @ y + b <= 0, # Linear boundaries are only allowed to intersect the sphere once
                (boundsA@Q.T)@y + boundsb <= 0
            ]
            prob = cp.Problem(cp.Minimize(cp.abs(y[0] + bi / np.linalg.norm(Ai))),constraints=lin_constraints)
            prob.solve(verbose=False,max_iters=1000)
            r = 0.0
            y = y.value
            
        # Now, finally we can sample a point on a disc with radius r around y
        # Project back and store
        for k in range(points_per_boundary):
            if y is None: # Solver might fail in the second sub problem. Then we can't sample.
                break
            if not do_sample[i] or (k > 0 and r == 0.0):
                break
                
            epsilon = np.random.randn(d)
            epsilon[0] = 0.0 # Don't change the first coordinate
            epsilon *= np.random.uniform()*r/np.linalg.norm(epsilon)
            x = Q.T@(y+epsilon)
            points.append(x)
            
        rs.append(r)
        mids.append(Q.T@y)
    return np.array(points), np.array(rs), mids


def sample_model(device, x_m_old, A, b, do_sample = None):
    """
    Perform line searches for the current model, and generate new datapoints (v_- and v_+ in the 
    main text; they're called x_m and x_p below).
    """
    # Sample points on the boundaries of the polytope
    points, rs, _ = sample_boundary_points(A, b, device.boundsA, device.boundsb, do_sample=do_sample)
    
    # Find center of polytope as point that has minimum distance to all points inside the polytope.
    midpoint = cp.Variable(x_m_old.shape[1])
    f = cp.mixed_norm(x_m_old - cp.reshape(midpoint,(1,x_m_old.shape[1])),2,1)
    cp.Problem(cp.Minimize(f)).solve(max_iters=1000)
    midpoint = midpoint.value
    r = np.min(np.linalg.norm(x_m_old-midpoint.reshape(1,-1),axis=1))
    
    # Make sure that the point we found is indeed inside the currently active polytope
    assert(device.inside_state(midpoint))
    
    x_m =[]
    x_p =[]
    # Conduct line-search starting from midpoint in the direction of the computed sample locations on the boundary
    for i in range(points.shape[0]):
        # Direction
        p = points[i,:] - midpoint
        
        # Line search and append to lists if we have a hit
        x_mi, x_pi, v = device.line_search(midpoint,p)
        if not v is None:
            x_m.append(x_mi)
            x_p.append(x_pi)
            
    return np.array(x_m), np.array(x_p), points.shape[0], rs


#SOLVER OF THE SUBPROBLEM

def log1pexp(x):
    return np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0) 

@primitive
def logsumexp(xs):
    """ Numerically more stable version of log of sum of exponentials """
    xmax = np.max(xs, axis = 1)
    expxs = np.exp(xs - xmax.reshape(-1,1))
    return np.log(np.sum(expxs,axis=1)) + xmax

def logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.reshape(g,(-1,1)) * np.exp(x - np.reshape(ans,(-1,1)))
defvjp(logsumexp, logsumexp_vjp)

class ConstantInteractionParameterisation:
    def __init__(self, Gamma, T):
        self.Gamma = Gamma
        self.T = T
        self.log_lambda_reg = 10.0
        self.log_lambda = np.zeros(Gamma.shape[0])
        self.s = np.zeros(T.shape[0])
        self.b = np.zeros(T.shape[0])
    
    def to_param(self):
        m = self.T.shape[0]
        n = self.T.shape[1]
        point = np.zeros(2*m+n+n*n)
        point[:n] = self.log_lambda
        point[n:n+m] = self.s
        point[n+m:n+m+m] = self.b
        
        return point
    
    def to_dict(self):
        return {'lambda':np.exp(self.log_lambda), 'c': log1pexp(self.s), 'b': self.b}
    
    def from_param(self,point):
        m = self.T.shape[0]
        n = self.T.shape[1]
        self.log_lambda = point[:n]
        self.s = point[n:n+m]
        self.b = point[n+m:n+m+m]
        
    def compute_planes(self):
        return np.diag(log1pexp(self.s)) @ self.T @ (np.diag(np.exp(self.log_lambda)) @ self.Gamma), self.b
        
    def regularizers(self):
        return self.log_lambda_reg*np.sum(self.log_lambda**2) # + self.eps_reg*np.sum(eps**2)

    def reset_transition(self, reset_markers, x_m, x_p, delta):
        self.s[reset_markers] = np.ones(np.sum(reset_markers))/delta
        A,_ = self.compute_planes()
        b0 = -np.max(A@x_m.T,axis=1)
        self.b[reset_markers] = b0[reset_markers]

def solve_max_likelihood_problem(x_m, x_p, parameterisation, delta, verbose = 0):

    def neg_log_loss(A, b, x, y):
        #linear activations
        act = x@A.T + b.reshape(1,-1)
        #log-nonlinear activations
        act = y * logsumexp(act)
        return log1pexp(-act)
        
    def obj(x):
        parameterisation.from_param(x)
        
        A,b = parameterisation.compute_planes()
        nll_m = neg_log_loss(A, b, x_m,-1)
        nll_p = neg_log_loss(A, b, x_p,1)
        
        # The regularizer
        penalty = 0.01*np.sum((A*delta)**2) + parameterisation.regularizers()
        
        # Compute loss
        return (np.sum(nll_m) + np.sum(nll_p) + penalty)/x_m.shape[0]
    
    x_init=parameterisation.to_param()
    
    # Solve
    res = minimize(obj, x_init, jac=grad(obj), tol=1.e-4, options={'disp':verbose > 1})
    parameterisation.from_param(res.x)
    return parameterisation
    
def count_separating(A, b, x_m, x_p, delta):
    """
    Given all of the facet normals (A) and their offsets (b), goes over all the points
    x_m and x_p (which should correspond to points below and above facets) and counts how 
    many of those pairs are indeed separated by a facet.
    """
    m = A.shape[0]
    d = A.shape[1]
    counts = np.zeros(m,dtype=np.int64)
    norms = np.linalg.norm(A,axis=1)
    
    # Count sample pairs that are separated by a plane
    f_m = (x_m @ A.T + b.reshape(1,-1))/np.maximum(norms.reshape(1,-1),1.e-5)
    f_p = (x_p @ A.T + b.reshape(1,-1))/np.maximum(norms.reshape(1,-1),1.e-5)
    separated = np.logical_and(f_m < 0, f_p > 0) # True if boundary passes between points
    
    for i in range(separated.shape[0]):
        # Could ignore points that are separated by multiple facets as they can be problematic?
        # if np.sum(separated[i]] == 1:
        counts[separated[i]] += 1
    
    return counts
    
    
#  The implementation of the full algorithm
def learn_convex_polytope(device, delta, startpoint, T, Gamma, max_searches = 15000, verbose = 0):
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

    # Sample initial dataset
    x_m, x_p, _ = naive_sampler(device, T.shape[1]**2,startpoint)
    
    d = Gamma.shape[1]
    num_searches = x_m.shape[0]
    counts = np.zeros(T.shape[0])
    params = ConstantInteractionParameterisation(Gamma, T)
    
    while True:
        renew_markers = counts < d+5
        params.reset_transition(renew_markers, x_m, x_p, delta)
        
        # Fit model
        params = solve_max_likelihood_problem(x_m, x_p, params, delta, verbose)
        
        A,b = params.compute_planes()
        # Check which facets we found
        found = np.linalg.norm(A,axis=1) > 0.1/delta
        
        # Sample new points, but include safe defaults for transitions that are not found
        params.reset_transition(~found, x_m, x_p, delta)
        A_norm,b_norm = params.compute_planes()
        
        do_sample = counts <= 2*(d+5)
        x_m_new, x_p_new, num_searches_iter, rs = sample_model(device, x_m, A_norm, b_norm, do_sample)
        if x_m_new.shape[0] == 0:
            x_m_new, x_p_new, _ = naive_sampler(device, 3*T.shape[0],np.mean(x_m,axis=0))
            num_searches_iter = x_m_new.shape[0]
            rs=np.ones(T.shape[0])
        x_m = np.vstack([x_m,x_m_new])
        x_p = np.vstack([x_p,x_p_new])
        x_m, x_p = filter_close(x_m,x_p, 0.5*delta)
        num_searches += num_searches_iter
        
        max_r_not_found = np.max(np.append(rs[~found],0.0))
        filter = np.logical_and(found, rs>2*delta)
        counts = count_separating(A, b, x_m, x_p, delta)
        
        if verbose > 1:
            print("Number of transitions found:", np.sum(filter))
            print("max_rad not found:", max_r_not_found)
            for pos in np.where(filter)[0]:
                print( T[pos,:], counts[pos], rs[pos])
                
            if verbose > 2:
                print("alphas:", np.exp(alpha)/np.exp(alpha[0]))
        
        #check, whether we are done
        if verbose > 0:
            print("Number of searches: ", num_searches,"/",max_searches)
            
        finished = False
        if np.sum(filter) > 0:
            finished = np.min(counts[filter])> (d+3) and max_r_not_found <= 2*delta
            
        if num_searches >= max_searches or finished:
            if verbose > 0:
                print("Finished learning polytope" if finished else "Max line searches exceeded")
                
                print("Number of transitions found:", np.sum(filter))
                print("max_rad not found:", max_r_not_found)
                for pos in np.where(filter)[0]:
                    print( T[pos,:], counts[pos], rs[pos])
                    
            return A, b, x_m, x_p, filter, num_searches, params.to_dict()
            
def generate_transitions_for_state(target_state, max_k=3, max_moves = None, has_reservoir=True):
    """
    Generates a list of all transitions that can be performed from 'target_state', based on 
    whether a reservoir is available, based on how many particles can move simultaneously (max_k),
    and based on how many moves they can make in total.
    
    This function works by first creating a list of all possible transitions, and then removing those
    that are not allowed by constraints. 
    """
    # Extract the number of dots
    n_dots = target_state.shape[0]
    
    Ts=[np.zeros((1,n_dots),dtype=np.int64)] # Adding starting condition for loop below. will be removed afterwards.
    
    # First we create a set of transitions by taking all transitions for max k-1 changes
    # and then add a few & unique 
    for k in range(max_k):
        
        prev=Ts[-1]
        Tk=[]
        for g in prev:      
            # For each dot ...
            for i in range(n_dots):
                # ... check if it is empty
                if g[i] == 0:
                    # If so, we create a new state with 1 particle on this dot
                    gnew = g.copy()
                    gnew[i] = 1
                    Tk.append(gnew.copy())
                    
                    # If the target state has at least one particle in this location already ...
                    if target_state[i] >= 1:
                        # ... also add the transition where this particle is removed
                        gnew[i] = -1
                        Tk.append(gnew)
                        
        # Keep only unique ones
        Tk = np.array(Tk) 
        Tk = np.unique(Tk,axis=0)
        
        if not len(Tk):
            continue
        
        # Filter transitions which are ruled out by max moves
        if not max_moves is None:
            keep_pos = np.sum(Tk == 1,axis=1) <= max_moves
            keep_neg = np.sum(Tk == -1,axis=1) <= max_moves
            Tk=Tk[np.logical_and(keep_pos,keep_neg)]

        Ts.append(Tk)
   
    # Now filter out all transitions, which are unphysical
    # i.e. more than one electron entering while none is leaving the array and vice versa
    for k in range(len(Ts)): #max_k+1):
        if k < 2:
            continue
        
        keep = np.abs(np.sum(Ts[k],axis=1)) < k
        Ts[k] = Ts[k][keep]
    
    # Select based on reservoir
    T=np.vstack(Ts[1:])
    if not has_reservoir:
        T=T[np.sum(T,axis=1)==0]
    return T
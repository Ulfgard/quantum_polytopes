from autograd import numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp
from scipy.optimize import minimize
import cvxpy as cp
from scipy.spatial.distance import cdist
from samplers import naive_sampler


def sample_boundary_points(A_full, b_full, boundsA, boundsb, points_per_boundary=3, do_sample = None):
    d = A_full.shape[1]
    #find points to sample
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

        #We have to sample a point on the facet of the constraint
        #to do this we compute the circle with maximum circumference inside the facet. For this we project 
        #the problem on the cut of the constraint with the remaining polytope. After we found the circle,
        #we can sample a random point in it and project it back into the original space
        #step 2.1: projection. we first rotate the coordinate constraint with a rotation Q so that Q@Ai->(|Ai|,0,...0)=|Ai|*e_1
        #we then know that since points must fullfill Ai@x + bi =0, the rotated points fulfill
        #Ai^T@Q^T@(Q@x)= Ai^T@Q^T@y= |Ai|*e_1^Ty = |Ai| y_1
        #and thus the constraint becomes y_1 = -bi/|Ai| which allows to project the dimension away
        
        #find Q
        #Q= I + [A_i; e_1 ] (R_theta - I)[A_i; e_1 ]^T, where R is a 2D rotation matrix with
        # theta=angle(A_i,e_i)
        # https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
        
        V=np.zeros((2,d))
        V[0,:] = Ai / np.linalg.norm(Ai)
        V[1,0] = 1
        V[1,:] -= V[0,0]*V[0,:]
        #check if we need to rotate at all
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

        
        #step 2.1:find the circle with maximum circumference inside the projected facet
        norm_vector = np.linalg.norm(A[:,1:], axis=1)
        
        r = cp.Variable(1)
        y = cp.Variable(d)
        lin_constraints = [
            y[0] == -bi / np.linalg.norm(Ai),#we stay on the ith constraint
            A @ y + b + r*norm_vector <= 0,  #linear boundaries are only allowed to intersect the sphere once
            (boundsA@Q.T)@y + r + boundsb <= 0, #also take bound constraints into account
            r >=0
        ]
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
        else:
            #transition does not intersect. Find minimum offset to boundary to make it intersect at at least a single point
            y = cp.Variable(d)
            lin_constraints = [
                A @ y + b <= 0, #linear boundaries are only allowed to intersect the sphere once
                (boundsA@Q.T)@y + boundsb <= 0
            ]
            prob = cp.Problem(cp.Minimize(cp.abs(y[0] + bi / np.linalg.norm(Ai))),constraints=lin_constraints)
            prob.solve(verbose=False,max_iters=1000)
            r = 0.0
            y = y.value
            
        #now, finally we can sample a point on a disc with radius r around y
        #project back and store
        for k in range(points_per_boundary):
            if y is None:
                break
            if not do_sample[i] or (k > 0 and r == 0.0):
                break
                
            epsilon = np.random.randn(d)
            epsilon[0] = 0.0 # don't change the first coordinate
            epsilon *= np.random.uniform()*r/np.linalg.norm(epsilon)
            x = Q.T@(y+epsilon)
            points.append(x)
        rs.append(r)
        mids.append(Q.T@y)
    return np.array(points), np.array(rs), mids


def sample_model(device, x_m_old, A, b, do_sample = None):
    points, rs, _ = sample_boundary_points(A, b, device.boundsA, device.boundsb, do_sample=do_sample)
    
    #find largest inscribed sphere
    midpoint = cp.Variable(x_m_old.shape[1])
    f=cp.mixed_norm(x_m_old - cp.reshape(midpoint,(1,x_m_old.shape[1])),2,1)
    cp.Problem(cp.Minimize(f)).solve(max_iters=1000)
    midpoint = midpoint.value
    r = np.min(np.linalg.norm(x_m_old-midpoint.reshape(1,-1),axis=1))
    assert(device.inside_state(midpoint))
    x_m =[]
    x_p =[]
    for i in range(points.shape[0]):
        p = points[i,:] - midpoint
        
        x_mi,x_pi, v = device.line_search(midpoint,p)
        if not v is None:
            x_m.append(x_mi)
            x_p.append(x_pi)
    x_m = np.array(x_m)
    x_p = np.array(x_p)
    return x_m, x_p, points.shape[0], rs


def log1pexp(x):
    return np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0) 

@primitive
def logsumexp(xs):
    xmax = np.max(xs, axis = 1)
    expxs = np.exp(xs - xmax.reshape(-1,1))
    return np.log(np.sum(expxs,axis=1)) + xmax
def logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.reshape(g,(-1,1)) * np.exp(x - np.reshape(ans,(-1,1)))
defvjp(logsumexp, logsumexp_vjp)

def compute_planes(alpha, s, G, eta):
    return np.diag(log1pexp(s)) @ G @ (np.diag(np.exp(alpha)) @ eta)
def solve_max_likelihood_problem(x_m, x_p, alpha_init, s_init, b_init, G, eta, delta):

    def neg_log_loss(A, b, x, y):
        
        #linear activations
        act = x@A.T + b.reshape(1,-1)
        
        #log-nonlinear activations
        act = y * logsumexp(act)
        #loss
        return log1pexp(-act)
        
    m = G.shape[0]
    n = G.shape[1]
    def obj(x):
        alpha = x[:n]
        s = x[n:n+m]
        b = x[n+m:]
        
        A =  compute_planes(alpha, s, G, eta)
        nll_m = neg_log_loss(A, b, x_m,-1)
        nll_p = neg_log_loss(A, b, x_p,1)
        
        penalty = 10*np.sum(alpha**2) + 0.01*np.sum((A*delta)**2)
        
        #compute loss
        return (np.sum(nll_m) + np.sum(nll_p) + penalty)/x_m.shape[0]
    x_init=np.zeros(2*m+n)
    x_init[:n] = alpha_init
    x_init[n:n+m] = s_init
    x_init[n+m:] = b_init
    res = minimize(obj, x_init, jac=grad(obj), tol=1.e-4, options={'disp':True})
    return res.x[:n], res.x[n:n+m], res.x[n+m:]
    
def stopping_criterion(A, b, x_m, x_p, delta):
    m = A.shape[0]
    d = A.shape[1]
    counts = np.zeros(m,dtype=np.int64)
    norms = np.linalg.norm(A,axis=1)
    
    #count sample pairs that are separated by a plane
    f_m = (x_m@A.T + b.reshape(1,-1))/np.maximum(norms.reshape(1,-1),1.e-5)
    f_p = (x_p@A.T + b.reshape(1,-1))/np.maximum(norms.reshape(1,-1),1.e-5)
    separated = np.logical_and(f_m < 0.5*delta, f_p > -0.5*delta) #true if boundary passes between points
    
    for i in range(separated.shape[0]):
    #ignore points that are separated by multiple facets as they can be problematic
    # if np.sum(separated[i]] == 1:
        counts[separated[i]] += 1
            
    #if there are d+1 close samples on all facets, we are done.
    # add  a few more samples just for the sake of precision
    stop = np.min(counts) >= d + 5
    return stop, counts, counts > (d + 5)
    
    
def generate_transitions_for_state(target_state, max_k=3, max_moves = None, has_reservoir=True):
    n_dots = target_state.shape[0]
    
    Gs=[np.zeros((1,n_dots),dtype=np.int64)] #adding starting condition for loop below. will be removed afterwards.
    #first create raw set of transitions
    #do this by taking all transitions for max k-1 changes
    #and then add a bew & unique 
    for k in range(max_k):
        prev=Gs[-1]
        Gk=[]
        for g in prev:
            for i in range(n_dots):
                if g[i] == 0:
                    gnew = g.copy()
                    gnew[i] = 1
                    Gk.append(gnew.copy())
                    if target_state[i] >= 1:
                        gnew[i] = -1
                        Gk.append(gnew)
        Gk = np.array(Gk) 
        Gk = np.unique(Gk,axis=0)
        
        #filter transitions which are ruled out by max moves
        if not max_moves is None:
            keep_pos = np.sum(Gk == 1,axis=1) <= max_moves
            keep_neg = np.sum(Gk == -1,axis=1) <= max_moves
            Gk=Gk[np.logical_and(keep_pos,keep_neg)]
        
        Gs.append(Gk)
    #now filter out all transitions, which are unphysical
    # i.e. more than one electron entering while none is leaving the array and vice versa
    for k in range(max_k+1):
        if k < 2:
            continue
        
        keep = np.abs(np.sum(Gs[k],axis=1)) < k
        Gs[k] = Gs[k][keep]
    
    #select base don reservoir
    G=[]
    for k in range(1,max_k+1):
        if k % 2 == 1 and not has_reservoir:
            continue
        G.append(Gs[k])
    G=np.vstack(G)
    return G
    
    
#an implementation of the full algorithm using state for easy future usage
def learn_convex_polytope(device, delta, startpoint, G, eta, max_searches = 15000):
    #function to filter close duplicates
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

    #sample initial dataset
    x_m, x_p, _ = naive_sampler(device, G.shape[1]**2,startpoint)
    
    d = eta.shape[1]
    alpha = np.zeros(eta.shape[0])
    num_searches = x_m.shape[0]
    counts = np.zeros(G.shape[0])
    s = np.zeros(G.shape[0])
    b = np.zeros(G.shape[0])
    while True:
        renew_idx = counts<d+5
        s[renew_idx] = np.ones(np.sum(renew_idx))/delta
        A = compute_planes(alpha, s, G, eta)
        b0 = -np.max(A@x_m.T,axis=1)#+0.5*delta*np.linalg.norm(A,axis=1)
        b[renew_idx] = b0[renew_idx]
        
        #check if alpha needs to be reset
        #we check whether we found a facet that includes an alpha and its label must have more than one electron location
        #multi_electron=(np.sum(G != 0,axis=1) > 1).reshape(-1,1)
        #found_facet = (~renew_idx).reshape(-1,1)
        #alpha_witness = np.logical_and(found_facet,multi_electron)*(G != 0)
        #reset_alpha = np.logical_and((np.sum(alpha_witness,axis=0) == 0), np.abs(alpha) > np.log(1.3))
        #alpha[reset_alpha] = 0.0
        
        #fit model
        alpha,s, b = solve_max_likelihood_problem(x_m, x_p, alpha, s, b, G, eta, delta)
        
        A = compute_planes(alpha, s, G, eta)
        found = np.linalg.norm(A,axis=1)> 0.1/delta
        
        #sample new points
        A_norm = compute_planes(alpha, np.zeros(G.shape[0]), G, eta)
        b_norm = -np.max(A_norm@x_m.T,axis=1)
        A_norm[found,:] = A[found,:]
        b_norm[found] = b[found]
        do_sample = counts <= 2*(d+5)
        x_m_new, x_p_new, num_searches_iter, rs = sample_model(device, x_m, A_norm, b_norm, do_sample)
        if x_m_new.shape[0] == 0:
            x_m_new, x_p_new, _ = naive_sampler(device, 3*G.shape[0],np.mean(x_m,axis=0))
            num_searches_iter = x_m_new.shape[0]
            rs=np.ones(G.shape[0])
        x_m = np.vstack([x_m,x_m_new])
        x_p = np.vstack([x_p,x_p_new])
        x_m, x_p = filter_close(x_m,x_p, 0.5*delta)
        num_searches += num_searches_iter
        
        max_r_not_found = np.max(np.append(rs[~found],0.0))
        filter = np.logical_and(found, rs>2*delta)
        _, counts, _ = stopping_criterion(A, b, x_m, x_p, delta)
        
        print("found:", np.sum(filter), "max_rad not found:", max_r_not_found)
        for pos in np.where(filter)[0]:
            print( G[pos,:], counts[pos], rs[pos])
        print("alphas:", np.exp(alpha)/np.exp(alpha[0]))
        
        #check, whether we are done
        #print(num_searches,"/",max_searches,x_m.shape[0])
        finished = False
        if np.sum(filter) > 0:
            finished = np.min(counts[filter])> (d+3) and max_r_not_found <= 2*delta
        if num_searches >= max_searches or finished:
            return A, b, x_m, x_p, filter, num_searches

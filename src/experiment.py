from autograd import numpy as np
from fit_zero_boundary import learn_zero_polytope
from fit_convex_polytope import learn_convex_polytope, generate_transitions_for_state, sample_boundary_points
import matplotlib.pyplot as plt
import pickle
import timeit

import sys
from simulator import sim_NxM, Simulator


experiment = sys.argv[1] #which problem to simulate
simid = int(sys.argv[2]) #id of this simulation. used for storage-ids. should be between 1 and 100
rho_int = int(sys.argv[3]) #inverse rho complexity parameter between 0 and 6
delta_int = int(sys.argv[4]) #inverse line search precision

delta = 1.0/delta_int
rho = rho_int/10.0


if experiment == "4x4nores": #S5
        n_rows = 4
        n_cols = 4
        has_reservoir = False
        target_states = np.ones((1,n_rows * n_cols))
        sim = sim_NxM(n_rows, n_cols, delta, rho)
        Ts = [generate_transitions_for_state(target_states[0], max_k=2, max_moves = 1, has_reservoir=has_reservoir)]
elif experiment == "4x4": #S4
        n_rows = 4
        n_cols = 4
        has_reservoir = True
        target_states = np.ones((1,n_rows * n_cols),dtype=np.int64)
        sim = sim_NxM(n_rows, n_cols, delta, rho)
        Ts = [generate_transitions_for_state(target_states[0], max_k=2, max_moves = 1, has_reservoir=has_reservoir)]
elif experiment == "3x3": #S3
        n_rows = 3
        n_cols = 3
        has_reservoir = True
        target_states = np.ones((1,n_rows * n_cols),dtype=np.int64)
        sim = sim_NxM(n_rows, n_cols, delta, rho)
        Ts = [generate_transitions_for_state(target_states[0], max_k=2, max_moves = 1)]        
elif experiment == "shuttle": #S1
        n_rows = 3
        n_cols = 2
        has_reservoir = True
        target_states = np.array([[1,0,0,0,1,0],[1,0,1,0,0,0],[1,0,0,1,0,0], [0,0,1,1,0,0], [0,0,0,1,1,0], [0,0,1,0,1,0]], dtype=np.int64)
        sim = sim_NxM(n_rows, n_cols, delta, rho)
        Ts = [generate_transitions_for_state(s, max_k=2, max_moves = 1) for s in target_states]
elif experiment == "ladder": #S2
        n_rows = 3
        n_cols = 2
        has_reservoir = True
        target_states = np.array([[1,0,0,1,1,0]], dtype=np.int64)
        sim = sim_NxM(n_rows, n_cols, delta, rho)
        Ts = [generate_transitions_for_state(target_states[0], max_k=4, max_moves=2)]
        Ts[0] = np.vstack([Ts[0],np.array([[0,1,1,-1,-1,1],[-1,0,1,-1,-1,1],[-1,1,0,-1,-1,1],[-1,1,1,0,-1,1],[-1,1,1,-1,0,1],[-1,1,1,-1,-1,0],[-1,1,1,-1,-1,1]],dtype=int)])
else:
    print("unknown experiment type")
    exit()

#filenames
filename_suffix="_"+experiment+"_"+str(simid)+"_"+str(rho_int)+"_"+str(delta_int)
filename_truth ="results/truth"+filename_suffix+".pkl"
filename_estimate ="results/estimate"+filename_suffix+".pkl"

#step 1: compute gamma/Gamma and prepare polytope
sim.set_reservoir(True)
zero_state = np.zeros(sim.num_dots,dtype=np.int64)
sim.activate_state(zero_state)
gammatruth = sim.boundaries(zero_state).A.copy()
print("optimal q:", -np.linalg.inv(gammatruth)@sim.boundaries(zero_state).b)
sim.set_reservoir(has_reservoir)

gamma,_,_, iter_info = learn_zero_polytope(sim,delta, num_start_samples = 4*sim.num_dots*(sim.num_dots+5))
gamma = gamma/np.linalg.norm(gamma[:,:-1],axis=1).reshape(-1,1)
gamma = gamma[:,:-1]

#find order
gamma_order = []
for i in range(sim.num_dots):
    jmax = -1
    maxv = -np.infty
    for j in range(sim.num_dots):
        if j in gamma_order: continue
        if np.abs(gamma[j,i]) > maxv:
            maxv = np.abs(gamma[j,i])
            jmax = j
    gamma_order.append(jmax)

gamma=gamma[gamma_order,:]
sys.stdout.flush()


results= []
truth=[]
for state_idx,target_state in enumerate(target_states):
    #setup simulation of this state
    sim_step = sim
    gamma_step = gamma
    #if we have no reservoir, we need to project away one degree of freedom to have a chance of closed polytopes
    if not has_reservoir:
        #there are N-1 independent directions formed by subtracting one of the principle directions from all other.
        #the Nth principle direction is
        v = np.linalg.inv(gamma_step)@np.ones(sim.num_dots)
        v /= np.linalg.norm(v)
        #it is easy to show that for any t with sum(t)=0:
        #t^TAv = 0

        #we remove the vector by defining a householder transformation that would move v->(1,0,...,0)
        #and then we remove the first element (we make use of the fact that the householder transformation is its own inverse)
        h = v.copy()
        h[0] -= 1.0
        h/=np.linalg.norm(h)
        P = np.eye(sim.num_dots) -2*np.outer(h,h)
        P = P[:,1:]
        sim_step = sim_step.slice(P,np.zeros(sim.num_dots))
        sim_step.set_reservoir(False)
        gamma_step = gamma_step@P
    sim_step.activate_state(target_state)
    
    #store ground truth
    polytope = sim_step.boundaries(target_state)
    _, radii, _ = sample_boundary_points(polytope.A, polytope.b, sim_step.boundsA, sim_step.boundsb)
    truth.append({
        "state":target_state.copy(), "Gamma": sim_step.Cinv@sim_step.C_g,
        "labels":polytope.labels.copy(), "A":polytope.A.copy(), "b":polytope.b.copy(),
        "radius":radii, "projection":np.ones(sim_step.num_dots,dtype=bool),
        "C": sim_step.C.copy(), "C_g": sim_step.C_g.copy(), "offset": sim_step.offset.copy(),
        "boundsA": sim_step.boundsA.copy(),"boundsb": sim_step.boundsb.copy() 
    })
    
    #compute a "bad" starting point
    sim_step.activate_state(target_state)
    startpoint, _, _ = sim_step.line_search(polytope.point_inside, np.random.randn(sim_step.num_inputs))
    startpoint = 0.95*startpoint + 0.05 * polytope.point_inside

    #print some info
    print("state:", target_state, polytope.point_inside)
    print("transitions_of_state:",polytope.A.shape[0])
    print(polytope.A)
    sys.stdout.flush()

    #step 2: fit the polytope
    print("rows T:", Ts[state_idx].shape[0])
    print(Ts[state_idx])
    start_time = timeit.default_timer()
    A,b,x_m,x_p,found, num_searches = learn_convex_polytope(sim_step, delta, startpoint, Ts[state_idx].astype(float), gamma_step, verbose=2)
    end_time = timeit.default_timer()
    
    #store results
    results.append({
        "state":target_state.copy(), "Gamma": gamma_step,
        "labels":Ts[state_idx].copy(), "A":A.copy(), "b":b.copy(),
        "found":found,
        "x_m":x_m, "x_p":x_p, "time":end_time-start_time,"searches":num_searches
    })
pickle.dump({"gamma":gammatruth, "polytopes":truth},open( filename_truth, "wb" ))
pickle.dump({"gamma":gamma, "polytopes":results, "gammainfo": iter_info},open( filename_estimate, "wb" ))

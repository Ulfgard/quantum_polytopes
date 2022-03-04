from functools import partial
import cvxpy as cp
import numpy as np

class Simulator:
    def __init__(self,C_g,C, delta, boundsA, boundsb, enumerator_func = None, verbose=False):
        #create matrices of constant interaction model
        self.has_reservoir = True
        self.C_g = C_g.copy()
        self.C = C.copy()
        self.Cinv = np.linalg.inv(self.C)
        self.num_inputs = C_g.shape[1]
        self.num_dots = C.shape[0]
        
        self.boundsA = boundsA
        self.boundsb = boundsb
        self.verbose = verbose
        
        #compute lower bound point
        lower_bound =cp.Variable(C_g.shape[1])
        prob = cp.Problem(cp.Minimize(np.ones(C_g.shape[1]) @ lower_bound),
             [self.boundsA @ lower_bound + self.boundsb <=0])
        
        prob.solve(verbose=False, max_iters=100000)
        self.lower_bound = lower_bound.value

        self.line_search_precision = delta
        self.prepared_transitions = {}
        self.active_polytope = None
        self.active_state = None
        self.n_dims = C_g.shape[1]
        self.enumerator_func = enumerator_func
        self.offset = np.zeros(self.C.shape[0])
    
    def _transition_equations(self, state_list, state_from):
        #compute normals
        A = (state_list - state_from) @ (self.Cinv@self.C_g)
        #compute offsets
        const_from = np.dot(state_from @ self.Cinv, state_from) 
        const_to = np.sum((state_list @ self.Cinv)*state_list,axis=1)
        b = (const_from - const_to)/2+(state_list - state_from) @ self.offset
        
        return A,b
            
    def _check_transition_existence(self, A, b):
        keep_list=np.ones(A.shape[0], dtype=bool)
        for k in range(A.shape[0]):
            #default assume it is not kept
            keep_list[k] = False
            Ak = A[keep_list,:] # take all previous selected or untested eqs, except the current
            bk = b[keep_list]
            A_eq = A[k]
            b_eq = b[k]
            c_lp = np.zeros(A.shape[1])
            try:
                x=cp.Variable(A.shape[1])
                prob = cp.Problem(cp.Minimize(c_lp @ x),
                     [A_eq @ x +b_eq == 0, Ak@ x + bk <= 0])
                prob.solve(verbose=False, max_iters=100000)
                if prob.status not in ["infeasible", "infeasible_inaccurate"]:
                    keep_list[k] = True
            except:
                continue
        return keep_list
        
    def _enumerate_state_batches(self, state_from):
        d = state_from.shape[0]
        if np.max(state_from) == 0:
            return [np.eye(d, dtype=int)+state_from]
        elif not (self.enumerator_func is None):
            return self.enumerator_func(state_from, self.has_reservoir)
        else:
            #compute list of possible state transitions for state_from
            #for simplicity, we assume that on a dot only a single electron can be added/subtracted
            #This leaves 3^d-1 states
            state_list=np.zeros((1,d),dtype=int)
            for i in range(d):
                state_list1 = state_list.copy()
                state_listm1 = state_list.copy()
                state_listm1[:,i] = -1
                state_list1[:,i] = 1
                state_list = np.vstack([state_list,state_list1])
                if state_from[i] >= 1:    
                    state_list = np.vstack([state_list,state_listm1])
            #first element is all-zeros
            state_list=state_list[1:]
            
            #without a reservoir, we only look at transitions that keep number of electrons constant.
            if not self.has_reservoir:
                state_list = state_list[np.sum(state_list,axis=1) == 0]
            
            return [state_list+state_from]
            
    def transitions(self, state_from):
        state_from = np.array(state_from, dtype=int)
        
        state_lists = self._enumerate_state_batches(state_from)

        As = []
        bs = []
        states = []
        for idx,state_list in enumerate(state_lists):
            A,b = self._transition_equations(state_list, state_from)

            keep_list=self._check_transition_existence(A,b)
            if self.verbose:
                print(idx," numel:", np.sum(keep_list),"/",state_list.shape[0])
            As.append(A[keep_list,:])
            bs.append(b[keep_list])
            states.append(state_list[keep_list])
            
        while True:        
            if len(As) == 1:
                return As[0], bs[0], states[0] - state_from
            else:
                #merge batches
                A = np.vstack(As[:2])
                b = np.concatenate(bs[:2])
                state = np.vstack(states[:2])
                #handle possible duplicate transitions
                state, indxs = np.unique(state,axis=0, return_index = True)
                A = A[indxs]
                b = b[indxs]
                if self.verbose:
                    print(len(As), "num elem after unique:", state.shape[0])
                
            
                #find transitions in the merged sets
                keep_list=self._check_transition_existence(A,b)
                #update lists
                As=As[2:]
                bs=bs[2:]
                states = states[2:]
                As.append(A[keep_list])
                bs.append(b[keep_list])
                states.append(state[keep_list])
    
    def _add_polytope(self, state, has_reservoir, A, b, labels, must_verify):
        state = state = np.array(state, dtype=int)
        #compute point inside
        norm_vector = np.linalg.norm(A, axis=1)
        r = cp.Variable(1)
        x = cp.Variable(self.C_g.shape[1])
        constraints = [
            A @ x + b + r * norm_vector <= 0, #linear boundaries are only allowed to intersect the sphere once
            self.boundsA @ x + r + self.boundsb <= 0, #also stay away from bound constraints
            r >=0
        ]
        prob = cp.Problem(cp.Maximize(r), constraints)
        prob.solve(verbose=False, max_iters=100000)
        
        polytope = type('', (), {})()
        polytope.A = A
        polytope.b = b
        polytope.labels = labels
        polytope.point_inside = x.value
        polytope.must_verify = must_verify
        
        #store polytope
        dict_key = tuple(state) + (has_reservoir,)
        self.prepared_transitions[dict_key] = polytope
        
    def precompute_polytopes(self, prepared_states):
        prepared_states = np.array(prepared_states, dtype=int)
        for state_from in prepared_states:
            dict_key = tuple(state_from) + (self.has_reservoir,)
            if dict_key in self.prepared_transitions.keys():
                #check, whether we were lazy before
                polytope = self.prepared_transitions[dict_key]
                if polytope.must_verify:
                    keep = self._check_transition_existence(polytope.A, polytope.b)
                    polytope.A = polytope.A[keep]
                    polytope.b = polytope.b[keep]
                    polytope.labels = polytope.labels[keep]
            else:
                #compute transitions            
                A,b, labels = self.transitions(state_from)
                self._add_polytope(state_from, self.has_reservoir, A,b,labels, False)
    
    def boundaries(self, state_from = None):
        if state_from is None:
            state_from = self.active_state
        state_from = np.array(state_from, dtype=int)
        self.precompute_polytopes(state_from.reshape(1,-1))
        
        return self.prepared_transitions[tuple(state_from)+(self.has_reservoir,)]
        
    def set_reservoir(self, use_reservoir):
        self.has_reservoir = use_reservoir
        
        if not self.active_state is None:
            self.activate_state(self.active_state)
        
    def activate_state(self, state):
       self.active_polytope = self.boundaries(state)
       self.active_state = np.array(state, dtype=int)
       
    
    
    def inside_state(self, p):
        f = self.active_polytope.A@p + self.active_polytope.b
        return np.all(f < 0)
        
        
    def line_search(self, x,p):

        #ensure starting point is inside the state
        y0 = self.active_polytope.A@x+self.active_polytope.b
        assert(np.all( y0 <= 0))
        p /= np.linalg.norm(p)
        #compute exact hit position on the polytope along the ray x+t*p, t>=0
        dir = self.active_polytope.A@p
        thit = -y0/dir
        if np.any(thit > 0):
            sel = np.where(thit > 0)[0]
            hit_id = sel[np.argmin(thit[sel])] #record which constraint got hit
            thit = np.min(thit[sel])
        else: #no constraint is hit
            thit = 1e100
            hit_id = None
        
        #clip maximum length according to bound constraints
        p0 = self.boundsA@p 
        if np.any(p0 > 0):
            sel = p0 > 0
            tmax = np.min(-(self.boundsb + (self.boundsA@x))[sel]/p0[sel])
            if tmax <= thit:
                hit_id = None
                thit = tmax
        #add noise and create bracket
        tm = thit - np.random.uniform(0,1.0)*self.line_search_precision
        tp = tm + self.line_search_precision
        return x+tm*p, x+tp*p, hit_id
        
    #compute slice via setting v= m+Px
    def slice(self,P,m):
        new_Cg = self.C_g@P
        new_offset = self.offset+self.Cinv@self.C_g@m
        
        new_boundsA = self.boundsA@P
        new_boundsb = self.boundsb + self.boundsA@m
        new_sim = Simulator(new_Cg, self.C, self.line_search_precision, new_boundsA, new_boundsb,self.enumerator_func)
        new_sim.offset = new_offset
        new_sim.has_reservoir = self.has_reservoir
        
        #copy over all existing polytopes and transform them. but be lazy, don't verify the equations, yet.
        for key in self.prepared_transitions.keys():
            polytope = self.prepared_transitions[key]
            A = polytope.A.copy()
            b = polytope.b.copy()
            labels = polytope.labels.copy()
            
            state = np.array(key,dtype=int)[:-1]
            b += A@m
            A = A@P
            has_reservoir = key[-1]
            
            new_sim._add_polytope(state, has_reservoir, A, b, labels, P.shape[0] != P.shape[1])
        new_sim.activate_state(self.active_state)
        return new_sim

def sim_NxM(rows, cols, delta, rho=0.1):
    n_dots = rows * cols
    def state_enumerator(state_from, has_reservoir):
        #first create set of transitions shared by all batches
        #this includes all single-electron transitions
        common_transitions = []
        for i in range(n_dots):
            state = np.zeros(n_dots)
            if has_reservoir:
                state[i] = 1
                common_transitions.append(state.copy())
                if state_from[i] >= 1:
                    common_transitions.append(-state)
            for j in range(i+1, n_dots):
                #transition i->j, j->i
                state = np.zeros(n_dots)
                state[i] = 1
                state[j] = -1
                if state_from[j] >= 1:  
                    common_transitions.append(state.copy())
                if state_from[i] >= 1:  
                    common_transitions.append(-state)
        common_transitions = np.array(common_transitions) 
        common_transitions = np.unique(common_transitions,axis=0)
        
        
        #all transitions in a maximum 3x3 block
        if cols <= 3:
            bcols = cols
            brows = min(rows, 9//bcols)
        else:
            brows = min(rows,3)
            bcols = min(cols, 9//brows)
        bdots = brows*bcols
        state_transitions=np.zeros((1,bdots),dtype=int)
        for i in range(bdots):
            state_transitions1 = state_transitions.copy()
            state_transitionsm1 = state_transitions.copy()
            state_transitionsm1[:,i] = -1
            state_transitions1[:,i] = 1
            state_transitions = np.vstack([state_transitions,state_transitions1])
            if state_from[i] >= 1:    
                state_transitions = np.vstack([state_transitions,state_transitionsm1])
        #first element is all-zeros
        state_transitions=state_transitions[1:]
        state_transitions = np.array(state_transitions)
        #filter impossible states when there is no reservoir
        if not has_reservoir:
            state_transitions = state_transitions[np.sum(state_transitions,axis=1) == 0]

        #filter duplicate states also available in common
        state_transitions = state_transitions[np.sum(np.abs(state_transitions),axis=1) > 2]
        
        #create the 3x3 batches
        for sx in range(max(1,rows - brows+1)):
            for sy in range(max(1,cols - bcols+1)):
                #copy state_transitions into the selected block
                #first find indexes of the target block
                indxs=[]
                for x in range(brows):
                    for y in range(bcols):
                        indxs.append((x+sx)*cols+(y+sy))
                transitions = np.zeros((state_transitions.shape[0],n_dots),dtype=int)
                transitions[:,indxs]= state_transitions
                #iterate over blocks of 500 and create batches
                transitions = np.array_split(transitions,transitions.shape[0]//500+1, axis = 0)
                #merge with common_transitions and save
                for batch in transitions:
                    yield np.vstack([batch,common_transitions]) + state_from
    
    e = 1.602* 1e-19
    unit = 1.e-18/e
    C_g  = np.zeros((rows,cols,rows,cols))
    
    C_0 = np.zeros((rows,cols,rows,cols))

    for i in range(rows):
        for j in range(cols):
            C_g[i,j,i,j] = 1
            if i < rows - 1:
                C_g[i,j,i+1,j] = rho
                C_g[i+1,j,i,j] = rho
            if j < cols - 1:
                C_g[i,j,i,j+1] = rho
                C_g[i,j+1,i,j] = rho
            if i < rows - 1 and j < cols - 1:
                C_g[i,j,i+1,j+1] = 0.3*rho
                C_g[i+1,j+1,i,j] = 0.3*rho
                C_g[i+1,j,i,j+1] = 0.3*rho
                C_g[i,j+1,i+1,j] = 0.3*rho
                
                
            if i < rows - 1:
                C_0[i,j,i+1,j] = rho
                C_0[i+1,j,i,j] = rho
            if j < cols - 1:
                C_0[i,j,i,j+1] = rho
                C_0[i,j+1,i,j] = rho
            if i < rows - 1 and j < cols - 1:
                C_0[i,j,i+1,j+1] = 0.3*rho
                C_0[i+1,j+1,i,j] = 0.3*rho
                C_0[i+1,j,i,j+1] = 0.3*rho
                C_0[i,j+1,i+1,j] = 0.3*rho
    C_g = unit * np.reshape(C_g,(n_dots, n_dots))
    C_0 = unit * np.reshape(C_0,(n_dots, n_dots))

    C_g = np.diag(np.exp(0.1*np.random.randn(n_dots))) @ C_g
    C_g += unit*0.02*np.random.rand(n_dots**2).reshape(n_dots, n_dots)
    gamma=np.exp(0.1*np.random.randn(n_dots))
    C_0 = np.diag(gamma) @ C_0 @ np.diag(gamma)
    Csum = np.sum(C_g,axis=1)+np.sum(C_0,axis=1)
    C = np.diag(Csum)-C_0
    return Simulator(C_g, C, delta, -np.eye(n_dots),-2*np.ones(n_dots), enumerator_func = state_enumerator)
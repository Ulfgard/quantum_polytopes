from functools import partial
import cvxpy as cp
import numpy as np

class Simulator:
    """
    This class simulates a quantum dot device using the constant interaction model, characterized by 
    the capacitance matrices C_g and C. 
    
    The simulator interally keeps track of the Coulomb diamonds (polytopes) and their transitions (facets),
    and takes care of keeping track of which transitions are feasible, with what precision, etc.
    This allows one to ask questions such as: "which transition does this facet correspond to?" and 
    "what is the orthogonal axis in voltage space (i.e. virtual gate) that tunes across it?". 
    The simulator will return, for each transition, a point on the transition line and the virtual gate.
    
    It also has the ability to take 2D slices through high dimensional voltage spaces to construct 2D 
    projections of charge stability diagrams.
    """
    
    def __init__(self, C_g, C, delta, boundsA, boundsb, enumerator_func = None, verbose=False):
        # Create the capacitance matrices of the constant interaction model
        self.has_reservoir = True
        self.C_g = C_g.copy()
        self.C = C.copy()
        self.Cinv = np.linalg.inv(self.C)
        
        # Extract the number of dots and gates
        self.num_inputs = C_g.shape[1]
        self.num_dots = C.shape[0]
        
        # Store the bounds for the polytope normals (A) and offsets (b)
        self.boundsA = boundsA
        self.boundsb = boundsb
        
        # Compute lower bound point
        lower_bound = cp.Variable(self.num_inputs)
        prob = cp.Problem(cp.Minimize(np.ones(C_g.shape[1]) @ lower_bound),
             [self.boundsA @ lower_bound + self.boundsb <=0])
        prob.solve(verbose=False, max_iters=100000)
        self.lower_bound = lower_bound.value

        # Set algorithm state and properties 
        self.line_search_precision = delta
        
        # Keep track of all the polytopes
        self.prepared_transitions = {}
        # The current polytope we're in
        self.active_polytope = None
        self.active_state = None
        
        # Store the function that enumerates transitions
        self.enumerator_func = enumerator_func
        self.offset = np.zeros(self.num_dots)
        
        # Output level
        self.verbose = verbose
    
    def _transition_equations(self, state_list, state_from):
        """
        Computes the normals and offsets for facets that separate 
        the state `state_from` and the states in `state_list`.
        """
        # Compute normals
        A = (state_list - state_from) @ (self.Cinv@self.C_g)
        # Compute offsets
        const_from = np.dot(state_from @ self.Cinv, state_from) 
        const_to = np.sum( (state_list @ self.Cinv)*state_list, axis=1)
        b = (const_from - const_to)/2 + (state_list - state_from) @ self.offset
        return A,b
            
    def _check_transition_existence(self, A, b):
        keep_list = np.ones(A.shape[0], dtype=bool)
        
        #check if there is any feasible point in the polytope
        #this can only happen, when computing a slice
        c_lp = np.zeros(A.shape[1])
        x = cp.Variable(A.shape[1])
        prob = cp.Problem(cp.Minimize(c_lp @ x), 
                [A@ x + b <= 0])
        prob.solve(verbose=False, max_iters=100000)
        if prob.status in ["infeasible", "infeasible_inaccurate"]:
            return None
        
        for k in range(A.shape[0]):
            # Default: assume it is not kept (could also initialize list with zeros)
            keep_list[k] = False
            
            Ak = A[keep_list,:] # take all previous selected or untested eqs, except the current
            bk = b[keep_list]
            A_eq = A[k]
            b_eq = b[k]
            
            try:
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
        
        # If this is the [0,0,...,0] state, we just want all the single electron states
        if np.max(state_from) == 0:
            return [np.eye(d, dtype=int)+state_from]
        elif not (self.enumerator_func is None): # Use the supplied enumerator to enumerate transitions
            return self.enumerator_func(state_from, self.has_reservoir)
        else:
            # Compute list of possible state transitions for state_from
            # For simplicity, we restrict to only single electron additions/subtractions per dot
            # This leaves 3^d-1 states
            state_list=np.zeros((1,d),dtype=int)
            for i in range(d):
                state_list1 = state_list.copy()
                state_listm1 = state_list.copy()
                state_listm1[:,i] = -1
                state_list1[:,i] = 1
                state_list = np.vstack([state_list,state_list1])
                if state_from[i] >= 1:    
                    state_list = np.vstack([state_list,state_listm1])
            
            # First element is all-zeros, we don't want it
            state_list=state_list[1:]
            
            # Without a reservoir, we only look at transitions that keep the number of electrons constant
            if not self.has_reservoir:
                state_list = state_list[np.sum(state_list,axis=1) == 0]
            
            return [state_list+state_from]
            
    def transitions(self, state_from):
        # Turn into numpy array of ints
        state_from = np.array(state_from, dtype=int)
        
        # Enumerate all states that can be reached from here
        state_lists = self._enumerate_state_batches(state_from)

        As = []
        bs = []
        states = []
        # Now for each of those, get the list of transitions...
        for idx,state_list in enumerate(state_lists):
            A,b = self._transition_equations(state_list, state_from)

            # ... and see if they are feasible
            keep_list=self._check_transition_existence(A,b)
            if keep_list is None:
                return np.zeros((0, self.num_inputs)),np.zeros(0), np.zeros(0)
            As.append(A[keep_list,:])
            bs.append(b[keep_list])
            states.append(state_list[keep_list])
            
        # Keep iterating over the list, merging facets, until they are all merged
        while True:
            if len(As) == 1:
                return As[0], bs[0], states[0] - state_from
            else:
                # Take the next set of As, bs and states, and merge them into a new set
                A = np.vstack(As[:2])
                b = np.concatenate(bs[:2])
                state = np.vstack(states[:2])
                # Update the lists; we've now taken care of another set of two
                As=As[2:]
                bs=bs[2:]
                states = states[2:]
                
                # Handle possible duplicate transitions
                state, indxs = np.unique(state, axis=0, return_index = True)
                A = A[indxs]
                b = b[indxs]
                # Find transitions in the merged sets
                keep_list = self._check_transition_existence(A,b)
                if keep_list is None:
                    return np.zeros((0, self.num_inputs)),np.zeros(0), np.zeros(0)

                # Add the merged ones back to the list
                As.append(A[keep_list])
                bs.append(b[keep_list])
                states.append(state[keep_list])
    
    def _add_polytope(self, state, has_reservoir, A, b, labels, must_verify):
        # Add another polytope we've found to the internal list of polytopes
        state = np.array(state, dtype=int)
        dict_key = tuple(state) + (has_reservoir,)
        
        #check if polytope is not existing
        if A.shape[0] == 0:
            polytope = type('', (), {})()
            polytope.A = A
            polytope.b = b
            polytope.labels = labels
            polytope.point_inside = None
            polytope.must_verify = False
            
            
            self.prepared_transitions[dict_key] = polytope
            return
    
        
        
        # Compute point inside
        norm_vector = np.linalg.norm(A, axis=1)
        r = cp.Variable(1)
        v = cp.Variable(self.num_inputs)
        constraints = [
            A @ v + b + r * norm_vector <= 0, #linear boundaries are only allowed to intersect the sphere once
            self.boundsA @ v + r + self.boundsb <= 0, #also stay away from bound constraints
            r >=0 # Radius is strictly positive
        ]
        
        # Find the largest possible inscribed hypersphere
        prob = cp.Problem(cp.Maximize(r), constraints)
        prob.solve(verbose=False, max_iters=100000)
        
        polytope = type('', (), {})()
        polytope.A = A
        polytope.b = b
        polytope.labels = labels
        polytope.point_inside = v.value
        polytope.must_verify = must_verify
        
        # Store polytope
        dict_key = tuple(state) + (has_reservoir,)
        self.prepared_transitions[dict_key] = polytope
        
    def precompute_polytopes(self, prepared_states):
        prepared_states = np.array(prepared_states, dtype=int)
        
        for state_from in prepared_states:
            # See if we already have this key in our prepared list
            dict_key = tuple(state_from) + (self.has_reservoir,)
            if dict_key in self.prepared_transitions.keys():
                # Check whether we were lazy before
                polytope = self.prepared_transitions[dict_key]
                if polytope.must_verify:
                    keep = self._check_transition_existence(polytope.A, polytope.b)
                    polytope.A = polytope.A[keep]
                    polytope.b = polytope.b[keep]
                    polytope.labels = polytope.labels[keep]
            else:
                # Compute transitions            
                A,b,labels = self.transitions(state_from)
                # Add the polytope described by these
                self._add_polytope(state_from, self.has_reservoir, A,b,labels, False)

    # Find the boundaries for the polytope that this state is in
    def boundaries(self, state_from = None):
        """
        Returns the normals and offsets for the facets/boundaries of the polygon that `state_from` is in.
        """
        if state_from is None:
            state_from = self.active_state
        # Convert to numpy array to be sure
        state_from = np.array(state_from, dtype=int)
        
        # Precompute polytopes (will check if already computed)
        self.precompute_polytopes(state_from.reshape(1,-1))
        
        # Keep track of having computed the transitions for this state
        dict_key = tuple(state_from) + (self.has_reservoir,)
        return self.prepared_transitions[dict_key]
        
    def set_reservoir(self, use_reservoir):
        self.has_reservoir = use_reservoir
        
        if not self.active_state is None:
            self.activate_state(self.active_state)
        
    def activate_state(self, state):
        """
        Sets the currently active state, and changes the active polytope to 
        the one this state is in.
        """
        self.active_state = np.array(state, dtype=int)
        self.active_polytope = self.boundaries(state)

    def inside_state(self, v):
        """
        Returns true if a point v is fully within the currently active polytope. 
        Excluding the boundary.
        """
        f = self.active_polytope.A@v + self.active_polytope.b
        return np.all(f < 0)
        
    def line_search(self, v, p):
        """
        Do a line search starting from voltage point v, in the direction p, until we hit
        the boundary of the active polytope
        """
        
        # Ensure starting point is inside the state
        y0 = self.active_polytope.A@v+self.active_polytope.b
        assert(np.all( y0 <= 0))
        
        # Normalize the ray direction
        p /= np.linalg.norm(p)
        
        # Compute exact hit position on the polytope along the ray v+t*p, t>=0
        d = self.active_polytope.A@p  # Project direction onto normal of active polytope
        thit = -y0/d
        
        if np.any(thit > 0):
            sel = np.where(thit > 0)[0]
            hit_id = sel[np.argmin(thit[sel])] # Record which constraint got hit
            thit = np.min(thit[sel])
        else: # No constraint is hit
            thit = 1e100
            hit_id = None
        
        # Clip maximum length according to bound constraints
        p0 = self.boundsA@p 
        if np.any(p0 > 0):
            sel = p0 > 0
            tmax = np.min(-(self.boundsb + (self.boundsA@v))[sel]/p0[sel])
            if tmax <= thit:
                hit_id = None
                thit = tmax

        # Add noise and create bracket (v_- and v_+)
        tm = thit - np.random.uniform(0,1.0)*self.line_search_precision
        tp = tm + self.line_search_precision
        return v+tm*p, v+tp*p, hit_id
        
    # Compute slice via setting v = m+Px
    def slice(self, P, m):
        # Deprecated! Newer version in the works
        new_Cg = self.C_g@P
        offset_div = self.Cinv@self.C_g@m
        new_offset = self.offset+offset_div
        
        new_boundsA = self.boundsA@P
        new_boundsb = self.boundsb + self.boundsA@m
        
        #throw out almost orthogonal bounds
        sel = np.linalg.norm(new_boundsA,axis=1)>1.e-7*np.linalg.norm(self.boundsA,axis=1)
        new_boundsA = new_boundsA[sel]
        new_boundsb = new_boundsb[sel]
        
        new_sim = Simulator(new_Cg, self.C, self.line_search_precision, new_boundsA, new_boundsb,self.enumerator_func)
        new_sim.offset = new_offset
        new_sim.has_reservoir = self.has_reservoir
        
        # Copy over all existing polytopes and transform them. 
        # But be lazy, don't verify the equations, yet.
        """
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
        """
        new_sim.activate_state(self.active_state)
        return new_sim

# Create an NxM array of quantum dots
def sim_NxM(rows, cols, delta, rho=0.1):
    n_dots = rows * cols
    
    def state_enumerator(state_from, has_reservoir):
        # First create set of transitions shared by all batches
        # This includes all single-electron transitions
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
        
        # Remove duplicates
        common_transitions = np.unique(np.array(common_transitions),axis=0)
             
        # All transitions in a maximum 3x3 block
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
    
    # Set units
    e = 1.602* 1e-19
    unit = 1.e-18/e
    # Initialize empty tensors for the capacitances
    C_g  = np.zeros((rows,cols,rows,cols))
    C_0 = np.zeros((rows,cols,rows,cols))

    # For each dot (x-coord)
    for i in range(rows):
        # For each dot (y-coord)
        for j in range(cols):
            # Diagonal capacitance = 1
            C_g[i,j,i,j] = 1
            
            # Capacitance to dot on the right and left
            if i < rows - 1:
                C_g[i,j,i+1,j] = rho
                C_g[i+1,j,i,j] = rho
            # Dot below
            if j < cols - 1:
                C_g[i,j,i,j+1] = rho
                C_g[i,j+1,i,j] = rho
            # Diagonal dots
            if i < rows - 1 and j < cols - 1:
                C_g[i,j,i+1,j+1] = 0.3*rho
                C_g[i+1,j+1,i,j] = 0.3*rho
                C_g[i+1,j,i,j+1] = 0.3*rho
                C_g[i,j+1,i+1,j] = 0.3*rho
                
            # Same as above
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
                
    # Reshape into n_dots x n_dots matrix
    C_g = unit * np.reshape(C_g,(n_dots, n_dots))
    C_0 = unit * np.reshape(C_0,(n_dots, n_dots))

    # (Matrix) multiply the diagonals by random numbers
    C_g = np.diag(np.exp(0.1*np.random.randn(n_dots))) @ C_g
    # Add random offsets to all elements
    C_g += unit*0.02*np.random.rand(n_dots**2).reshape(n_dots, n_dots)
    
    # Multiply C_0 by random numbers too
    gamma=np.exp(0.1*np.random.randn(n_dots))
    C_0 = np.diag(gamma) @ C_0 @ np.diag(gamma)
    
    # Compute the total capacitcance per col
    Csum = np.sum(C_g,axis=1)+np.sum(C_0,axis=1)
    C = np.diag(Csum)-C_0
    
    # Simulator(C_g, C, delta, boundsA, boundsb, enumerator_func = None, verbose=False):
    return Simulator(C_g, C, delta, -np.eye(n_dots), -2*np.ones(n_dots), enumerator_func = state_enumerator)
import pickle
import numpy as np
import itertools
from collections import defaultdict
from fit_convex_polytope import sample_boundary_points


class Tree(defaultdict):
    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)

bases = ["ladder", "shuttle", "3x3", "4x4","4x4nores"]
base_info = {}
base_info["3x3"] = [9,9,[]]
base_info["4x4to3x3"] = [16,9,[]]
base_info["4x4nores"] = [16,15,[]]
base_info["4x4vnores"] = [16,15,[]]
base_info["4x4"] = [16,16,[]]
base_info["shuttle"] = [6,6]
base_info["ladder"] = [6,6]
rhos =[3,1]
deltas = [500,1000]

all_experiments = itertools.product(bases,rhos,deltas)
results_virtual = Tree()
results_poly = Tree()
for base,rho,delta in all_experiments:
    print(base,rho,delta)
    dim_base = base_info[base][0]
    dim_sub = base_info[base][1]

    
    err_A0 = np.pi*np.ones((20,dim_base))
    err_virt = np.pi*np.ones((20,dim_base))
    A0success = np.ones(20,dtype=bool)
    A0searches = np.ones(20)
    for run in range(20):
        
        file_truth = "results/"+base+"/truth_"+base+"_"+str(run+1)+"_"+str(rho)+"_"+str(delta)+".pkl"
        file_res = "results/"+base+"/estimate_"+base+"_"+str(run+1)+"_"+str(rho)+"_"+str(delta)+".pkl"
        try:
            with open(file_truth, "rb") as f:
                data_truth = pickle.load(f)
            with open(file_res, "rb") as f:
                data_res = pickle.load(f)
        except:
            print("error loading run "+file_truth+". Skipping.")
            A0success[run] = False
            continue     
        #evaluate A0
        A0_truth = data_truth['A0']
        A0_res = data_res['A0']
        A0_truth /= np.linalg.norm(A0_truth,axis=1).reshape(-1,1)
        A0_res /= np.linalg.norm(A0_res,axis=1).reshape(-1,1)
        
        virt_coords = A0_truth@np.linalg.pinv(A0_res)
        virt_coords /= np.linalg.norm(virt_coords,axis=1).reshape(-1,1)
        
        err_A0[run] = np.arccos(np.diag(A0_truth@A0_res.T))
        err_virt[run] = np.arccos(np.diag(virt_coords))
        #evaluate success
        A0matches = np.sum(data_res['A0info'][-1][1] >=dim_base+5)
        A0success[run] = (A0matches == dim_base)
        
        A0searches[run] = data_res['A0info'][-1][0]
        
        #no need to continue, if step 1 failed
        if not A0success[run]:
            continue

        #continue measuring the estimate of the polytope
        results_poly[base][rho][delta][run]=[]
        for ind_p,p_truth,p_res in zip(range(len(data_truth['polytopes'])), data_truth['polytopes'],data_res['polytopes']):
            #load polytope equations and normalize
            A_truth = p_truth['A']
            A_res = p_res['A']
            b_truth = p_truth['b']
            b_res  = p_res['b']
            b_truth /= np.linalg.norm(A_truth,axis=1)
            b_res /= np.linalg.norm(A_res,axis=1)+1.e-5
            A_truth /= np.linalg.norm(A_truth,axis=1).reshape(-1,1)
            A_res /= np.linalg.norm(A_res,axis=1).reshape(-1,1)+1.e-5
            
            
            radius_truth = p_truth['radius']
            labels_truth = p_truth['labels']
            T = p_res['labels']
            found_res = p_res['found']
            
            #obtain geometry information of learned polytope
            boundsA=-np.eye(dim_sub)
            boundsb = -2*np.ones(dim_sub)
            radius_res=np.zeros(A_res.shape[0])
            mid_res = np.zeros((A_res.shape[0],dim_sub))
            _, radius_res[found_res], mid_res[found_res] = sample_boundary_points(A_res[found_res], b_res[found_res], boundsA, boundsb)
            point_inside = np.mean(mid_res[found_res],axis=0)
            found_res = (radius_res > 2.0/delta)
            num_false_pos = 0
            num_interesting = 0
            num_found = 0
            err_angle = []
            err_b = []
            radius_interesting = []
            found_interesting = []
            workable_interesting = []
            label_interesting = []
            
            #enumerate all facets we can possibly find with our algorithm
            for pos_res,t in enumerate(T):
                diff = np.sum(np.abs(labels_truth - t.reshape(1,-1)),axis=1)
                if np.all(diff) > 1.e-3:
                    #facet does not exist, so check for false positive
                    num_false_pos += found_res[pos_res]
                    continue
                pos_truth = np.where(diff < 1.e-3)[0][0]
                #check if the facet is large enough to be interesting:
                if radius_truth[pos_truth] < 2.0/delta: #selection criterion
                    continue
                #facet does exist in this polytope, we search for it and is large enough, so evaluate
                num_interesting += 1
                label_interesting.append(t)
                radius_interesting.append(radius_truth[pos_truth])
                found_interesting.append(found_res[pos_res])
                num_found += found_res[pos_res]
                workable_interesting.append(False) #to be updated later
                #no need to continue if we have not found it
                if not found_res[pos_res]:
                    continue
                
                #measure errors in parameters
                err_angle.append(np.arccos(A_truth[pos_truth]@A_res[pos_res]))
                err_b.append(np.abs(b_truth[pos_truth]-b_res[pos_res]))
                
                #check whether the transition is workable
                direction = mid_res[pos_res]-point_inside
                direction /= np.linalg.norm(direction)
                
                A_line = A_truth @ direction
                b_line = b_truth + A_truth @ point_inside
                positive = np.where(A_line > 0)[0]
                index = np.argmin(-b_line[positive]/A_line[positive])
                if positive[index] == pos_truth:
                    workable_interesting[-1] = True
                else:
                    print("found, but non-workable:", t)
                
            
            if num_interesting == 0:
                results_poly[base][rho][delta][run].append({"num_found":0, "num_interesting":0,
                "radius":[], "found":[],
                "err_angle":[],
                "err_b":[],
                "false_pos": num_false_pos, "time":0, "searches":0
            })
            print(num_found, num_interesting, num_false_pos)
            radius_interesting=np.array(radius_interesting)
            found_interesting=np.array(found_interesting,dtype=bool)
            workable_interesting = np.array(workable_interesting, dtype=bool)
            if num_false_pos > 0:
                print("false positives!: ", num_false_pos)
            results_poly[base][rho][delta][run].append({
                "num_found":num_found, "num_interesting":num_interesting, "num_workable":np.sum(workable_interesting),
                "labels": label_interesting, "radius":radius_interesting, "found":found_interesting, "workable": workable_interesting,
                "err_angle":err_angle,
                "err_b":err_b,
                "false_pos": num_false_pos,
                "time": p_res["time"], "searches":p_res["searches"]
            })
    #store all virtual gate results
    results_virtual[base][rho][delta]={
        "success":np.sum(A0success),
        "error_A0":np.quantile(err_A0[A0success],[0,0.05,0.5,0.95,1]),
        "error_virt":np.quantile(err_virt[A0success],[0,0.05,0.5,0.95,1]),
        "searches":np.quantile(A0searches[A0success],[0,0.05,0.5,0.95,1])
    }

    print(results_virtual[base][rho][delta]["success"], results_virtual[base][rho][delta]["error_A0"], results_virtual[base][rho][delta]["error_virt"])

with open("results/poly_eval.pkl","wb") as f:
    pickle.dump(results_poly, f)

with open("results/virtual_eval.pkl","wb") as f:
    pickle.dump(results_virtual, f)
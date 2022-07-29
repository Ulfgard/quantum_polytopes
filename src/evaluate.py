import numpy as np
from matplotlib import pyplot as plt
import pickle
from collections import defaultdict

class Tree(defaultdict):
    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)

with open("results/virtual_eval.pkl", "rb") as f:
    results_virtual = pickle.load(f)

with open("results/poly_eval.pkl", "rb") as f:
    results_poly = pickle.load(f)

bases_virt=["ladder", "3x3", "4x4"]
names_virt= {"shuttle":"$3\\times2$","ladder":"$3\\times2$", "3x3":"$3\\times3$", "4x4":"$4\\times4$"}
bases_poly=["shuttle", "ladder", "3x3", "4x4", "4x4fast", "4x4nores"]
names_poly= {"shuttle":"$S_1$","ladder":"$S_2$", "3x3":"$S_3$", "4x4":"$S_4$", "4x4fast":"$S_5$", "4x4nores":"$S_6$"}
markers = ('x', '+', '.', '1', '*','d')
colors=('r','orange','blue','black', 'grey', 'green')
settings= [(1,1,500),(2,1,1000),(3,3,500),(4,3,1000)]
#num unsuccessful
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_virt):
    successes=np.zeros(4)
    for j,rho,delta in settings:
        successes[j-1] = results_virtual[base][rho][delta]["success"]/20
    plt.plot(successes,[1,2,3,4], markers[i], label=names_virt[base], c=colors[i])

plt.gca().set_yticks([1,2,3,4])
plt.gca().set_xticks([0.0,0.5,1.0])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_xlim(-0.05,1.05)
plt.gca().set_ylim(0.5,4.5)
plt.legend()
plt.xlabel("Fraction Successful Trials")
plt.tight_layout()
plt.savefig("results/zero_num_successful.png")
plt.savefig("results/zero_num_successful.eps",format="eps")
plt.close()


#angle error
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_virt):
    for j,rho,delta in settings:
        label = None
        if j == 1:
            label = names_virt[base]
        qs = results_virtual[base][rho][delta]["error_A0"]*180/np.pi
        y=j+(i/len(bases_virt)-0.5)*0.7
        plt.plot(qs[[1,3]],(y,y), markers[i]+'--', label=label, c=colors[i],linewidth=0.5)
        plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_ylim(0.5,4.5)
plt.legend()
plt.xlabel("Deviation of Normals (Degrees)")
plt.tight_layout()
plt.savefig("results/zero_angle_err.png")
plt.savefig("results/zero_angle_err.eps",format="eps")
plt.close()

plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_virt):
    for j,rho,delta in settings:
        label = None
        if j == 1:
            label = names_virt[base]
        qs = results_virtual[base][rho][delta]["error_virt"]*180/np.pi
        y=j+(i/len(bases_virt)-0.5)*0.7
        plt.plot(qs[[1,3]],(y,y), markers[i]+'--', label=label, c=colors[i],linewidth=0.5)
        plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_ylim(0.5,4.5)
plt.legend()
plt.xlabel("Deviation of Normals (Degrees)")
plt.tight_layout()
plt.savefig("results/zero_vangle_err.png")
plt.savefig("results/zero_vangle_err.eps",format="eps")
plt.close()

#searches
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_virt):
    for j,rho,delta in settings:
        qs = results_virtual[base][rho][delta]["searches"]
        y=j+(i/len(bases_virt)-0.5)*0.7
        label = None
        if j == 1:
            label = names_virt[base]
        plt.plot(qs[[1,3]],(y,y), markers[i]+'--', label=label, c=colors[i],linewidth=0.5)
        #plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
        
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_ylim(0.5,4.5)
plt.gca().set_xlim(100,10000)
plt.xscale('log')
plt.gca().set_xticks([100,1000,10000])
plt.gca().set_xticklabels([r"100",r"1000",r"10000"])
#plt.legend()
plt.xlabel("Number of Line-searches")
plt.tight_layout()
plt.savefig("results/zero_searches.png")
plt.savefig("results/zero_searches.eps",format="eps")
plt.close()

#fraction interesting found
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_poly):
    
    num_interesting = np.zeros(4)
    num_found = np.zeros(4)
    for j,rho,delta in settings:
        for run in results_poly[base][rho][delta]:
            for poly in results_poly[base][rho][delta][run]:
                num_interesting[j-1] += poly["num_interesting"]
                num_found[j-1] += poly["num_found"]
    plt.plot(num_found/num_interesting,[1,2,3,4], markers[i], label=names_poly[base], c=colors[i])
    print(base, num_interesting - num_found, num_interesting, 100*(1-num_found/num_interesting))
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_xticks([0.9,0.95,1.0])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_xlim(0.8905,1.005)
plt.gca().set_ylim(0.5,4.5)
plt.xlabel("Fraction Found")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("results/poly_num_found.png")
plt.savefig("results/poly_num_found.eps",format="eps")
plt.close()


#fraction found that are workable
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_poly):
    
    num_workable = np.zeros(4)
    num_found = np.zeros(4)
    for j,rho,delta in settings:
        for run in results_poly[base][rho][delta]:
            for poly in results_poly[base][rho][delta][run]:
                num_workable[j-1] += poly["num_workable"]
                num_found[j-1] += poly["num_found"]
    
    plt.plot(num_workable/num_found,[1,2,3,4], markers[i], label=names_poly[base], c=colors[i])

plt.gca().set_yticks([1,2,3,4])
plt.gca().set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_xlim(0.45,1.05)
plt.gca().set_ylim(0.5,4.5)
plt.legend()
plt.xlabel("Fraction Found")
plt.tight_layout()
plt.savefig("results/poly_num_workable.png")
plt.savefig("results/poly_num_workable.eps",format="eps")
plt.close()

#confidence intervals of radii
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_poly):
    for j,rho,delta in settings:
        
        label = None
        if j == 1:
            label = names_poly[base]
        radii=np.zeros(0)
        for run in results_poly[base][rho][delta]:
            for poly in results_poly[base][rho][delta][run]:
                radii = np.append(radii, poly["radius"][~poly["found"]]*delta)
        if radii.shape[0] <= 1:
            qs = [2.02,2.02,2.02]
        else:
            qs = np.quantile(np.maximum(radii,2.02),[0.5,0.95,1])
        y=j+0*(i/len(bases_virt)-0.5)*0.7
        plt.plot(qs[1]/2-1,y, markers[i], label=label, c=colors[i],linewidth=0.5)
        plt.plot(qs[0]/2-1,y, markers[i], c=colors[i],linewidth=0.5)
        #plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.xscale('log')
plt.gca().set_xticks([0.01,0.1,1,10])
plt.gca().set_xlim(0.007,100)
plt.gca().set_xticklabels(["<1%", "10%", "100%", "1000%"])
plt.gca().set_ylim(0.5,4.5)
plt.xlabel("$\\frac{r}{r_{\min}}-1$ in percent")
plt.legend()
plt.tight_layout()
plt.savefig("results/poly_radii_missed.png")
plt.savefig("results/poly_radii_missed.eps",format="eps")
plt.close()

#runtime
print ("runtime")
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_poly):
    for j,rho,delta in settings:
        times = []
        for run in results_poly[base][rho][delta]:
            for poly in results_poly[base][rho][delta][run]:
                times.append(poly["time"])
        qs = np.quantile(times,[0,0.05,0.5,0.95,1])
        y=j+(i/len(bases_poly)-0.5)*0.7
        label = None
        if j == 1:
            label = names_poly[base]
        plt.plot(qs[[1,3]],(y,y), markers[i]+'--', label=label, c=colors[i],linewidth=0.5)
        print(base, rho, delta, qs[2])
        #plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
        
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_ylim(0.5,4.5)
plt.xscale('log')
#plt.legend()
plt.xlabel("Total running time (s)")
plt.tight_layout()
plt.savefig("results/poly_time.png")
plt.savefig("results/poly_time.eps",format="eps")
plt.close()

#searches
print ("line searches")
plt.figure(figsize=(4,2.5))
for i,base in enumerate(bases_poly):
    for j,rho,delta in settings:
        searches = []
        for run in results_poly[base][rho][delta]:
            for poly in results_poly[base][rho][delta][run]:
                searches.append(poly["searches"])
        qs = np.quantile(searches,[0,0.05,0.5,0.95,1])
        y=j+(i/len(bases_poly)-0.5)*0.7
        label = None
        if j == 1:
            label = names_poly[base]
        print(base, rho, delta, qs[2])
        plt.plot(qs[[1,3]],(y,y), markers[i]+'--', label=label, c=colors[i],linewidth=0.5)
        #plt.plot(qs[2],y, markers[i], c=colors[i],linewidth=0.5)
        
plt.gca().set_yticks([1,2,3,4])
plt.gca().set_yticklabels([r"$\rho=1,\delta=0.002$",r"$\rho=1, \delta=0.001$",r"$\rho=3, \delta=0.002$",r"$\rho=3, \delta=0.001$"])
plt.gca().set_ylim(0.5,4.5)
plt.gca().set_xlim(100,15000)
plt.xscale('log')
plt.gca().set_xticks([100,1000,10000])
plt.gca().set_xticklabels([r"100",r"1000",r"10000"])
#plt.legend()
plt.xlabel("Number of Line-searches")
plt.tight_layout()
plt.savefig("results/poly_searches.png")
plt.savefig("results/poly_searches.eps",format="eps")
plt.close()

#check shuttle
shuttle_ops=np.array([
    [0,0,1,0,-1,0],
    [0,0,-1,1,0,0],
    [-1,0,1,0,0,0],
    [0,0,-1,0,1,0],
    [0,0,1,-1,0,0],
    [1,0,-1,0,0,0]
])
shuttle_ops_prev=-np.array([
    [1,0,-1,0,0,0],
    [0,0,1,0,-1,0],
    [0,0,-1,1,0,0],
    [-1,0,1,0,0,0],
    [0,0,-1,0,1,0],
    [0,0,1,-1,0,0]
])
shuttle_ind_prev=[5,0,1,2,3,4]
for j,rho,delta in settings:
    base="shuttle"
    solved = 0
    possible = 0
    for run in results_poly[base][rho][delta]:
        workable=np.zeros(6)
        num_exist = 0
        for ind,poly in enumerate(results_poly[base][rho][delta][run]):
            t = shuttle_ops[ind]
            t_prev = shuttle_ops_prev[ind]
            
            diff = np.sum(np.abs(poly["labels"] - t.reshape(1,-1)),axis=1)
            pos = np.argmin(diff)
            diff_prev = np.sum(np.abs(poly["labels"] - t_prev.reshape(1,-1)),axis=1)
            pos_prev = np.argmin(diff)
            if diff[pos] <1.e-3:
                num_exist += 1
                workable[ind] += poly["workable"][pos]
            if diff[pos_prev] <1.e-3:
                workable[shuttle_ind_prev[ind]] += poly["workable"][pos_prev]
                
        num_workable = np.sum(workable > 0)
        if num_exist == 6:
            possible += 1
            if num_workable == 6:
                solved += 1
        else:
            print("not all required facets exists/are large enough")
    print("num shuttle solved:", rho, delta, solved, possible)

t=np.array([-1,1,1,-1,-1,1])
for j,rho,delta in settings:
    solved = 0
    possible = 0
    for run in results_poly["ladder"][rho][delta]:
        poly = results_poly["ladder"][rho][delta][run][0]
        diff = np.sum(np.abs(poly["labels"] - t.reshape(1,-1)),axis=1)
        pos = np.argmin(diff)
        if diff[pos] <1.e-3:
            possible += 1
            solved += poly["workable"][pos]
        
    print("num ladder solved:", rho, delta, solved, possible)

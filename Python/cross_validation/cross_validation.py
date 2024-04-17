import torch
import numpy as np 
import pandas as pd
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
from Python.cross_validation.cross_valid_score import *
import csv 
import sys 
import os
from functions_optim import *



if __name__ == '__main__':

    data = pd.read_csv("data/subtrees_cpm_cpm.csv")
    tree_list=[]
    data.columns=[str(k) for k in range(len(data.columns))]
    f=1.
    tot_time=max(torch.from_numpy(np.array(data[str(1)].dropna().values[1:],dtype=np.float64))).item()
    #Time=torch.logspace(0,np.log(tot_time+1.5)/np.log(10),300)-1
    Time=torch.linspace(0.0,tot_time,300)

    for k in range(len(data.columns)):
        if data[str(k)][0]=="crown" :
            T_b=torch.from_numpy(np.array(data[str(k)].dropna().values[1:],dtype=np.float64))
            #Time= torch.unique(torch.cat([T_b,torch.logspace(0,np.log(T_b[-1]+1)/np.log(10),n_times-len(T_b))-1]),sorted=True)
            tree_list.append(tree(T_b,Time,f,data[str(k)].values[0],None))
        elif data[str(k)][0]=="stem" :
            T_b=torch.from_numpy(np.array(data[str(k)].dropna().values[1:-1],dtype=np.float64))
            #Time= torch.unique(torch.cat([T_b,torch.logspace(0,np.log(T_b[-1]+1)/np.log(10),n_times-len(T_b))-1]),sorted=True)
            tree_list.append(tree(T_b,Time,f,data[str(k)].values[0],float(data[str(k)].dropna().values[-1])))
    


    print("data loaded")
    

    ### parameters ###
    pen_l=["cpm",1]
    pen_m=["cpm",1]
    statistic="LTT"
    ## Initialzation for the optimization 

    n_iter=200

    ## Learning rate 
    lr0=0.05
    B=len(tree_list)
    

    tree_test=[tree_list[i] for i in range(B) ]
    tree_learn=[ [tree_list[k] for k in range(len(tree_list)) if k!=i] for i in range(B)]
    a_init=torch.full_like(Time, np.log(0.8))
    b_init=torch.full_like(Time, np.log(0.5))
    optim_init=estim_param(a_init,b_init,n_iter,lr0)

    beta_max=100.
    alpha_max=100.
    n_grid=10
    alpha_grid=torch.tensor([alpha_max/(2**i) for i in range(n_grid)]) 
    beta_grid=torch.tensor([beta_max/(2**i) for i in range(n_grid)]) 
    arguments=[ (optim_init, tree_learn, tree_test ,alpha_grid[i],beta_grid[j],pen_l,pen_m, len(Time),statistic) for i in range(n_grid) for j in range(n_grid)]
    num_processes = len(arguments) #number of processes to use for parallelization

    with mp.Pool(num_processes) as pool:
        results = pool.starmap(score_parra, arguments)

    # Close the pool and wait for the worker processes to finish
    pool.close()
    pool.join()
        

    
    np.savetxt('Matrix_score_sim_LTT_cpm.csv', results ,delimiter=',')

    
    print("score computed")
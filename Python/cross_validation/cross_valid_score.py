import torch
import numpy as np 
import pandas as pd
import torch.multiprocessing as mp
import csv 
import sys
# Add the path to the folder containing your module
sys.path.append('/Users/jeanv./Desktop/stage-cladogenese/Code/Code-python/')

# Import the module
from functions_optim import *


# Compute the loss for a list of trees

def loss_multiple_trees(l,m,tree,pen,statistic): 
    loss=0
    for i in range(len(tree)):
        loss += objective(l,m,tree[i],pen,statistic)
    return loss

def optim_multiple_trees(estim_param, trees, penalties,statistic, eps=0.001, threshold=0.01, patience=3):
    mses=[] #to store the evolution of the objective function during the gradient descent
    a=estim_param.a_init.clone().detach().requires_grad_(True)
    b=estim_param.b_init.clone().detach().requires_grad_(True)
    grad_list=[]
    a_list=[a.clone().detach()]
    b_list=[b.clone().detach()]
    stop=False
    count=0
    epoch=0
    assert statistic in ["likelihood","LTT"]
    if estim_param.opt=='adam':
        optimizer = torch.optim.Adam([a,b], lr=estim_param.lr0, betas=(0.9, 0.9), eps=1e-08, weight_decay=0, amsgrad=False)
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',threshold=threshold,factor=0.9, patience=5,verbose=True) #learning rate scheduler

    elif estim_param.opt=='sgd':
        optimizer = torch.optim.SGD([a,b], lr=estim_param.lr0,momentum=0.9,nesterov=True)
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',threshold=threshold,factor=0.7, patience=4,verbose=True) #learning rate scheduler

    elif estim_param.opt=='adagrad':
        optimizer = torch.optim.Adagrad([a,b], lr=estim_param.lr0)



    for i in range(estim_param.n_iter):
        epoch+=1
        loss=loss_multiple_trees(torch.exp(a),torch.exp(b),trees,penalties,statistic)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)

        a_list.append(a.clone().detach())
        b_list.append(b.clone().detach())

        mses.append(loss.item())

        grad_norm=torch.norm(torch.cat([a.grad,b.grad]))
        grad_list.append(grad_norm.item())


        #stopping condition : 
        diff=torch.norm(torch.cat([a_list[-1]-a_list[-2], b_list[-1]-b_list[-2] ]))

        if diff < eps :
            count +=1
            print("diff : ", diff)
            print("count : ", count)
        else :
            count=0

        if count > patience :
            print("convergence reached")
            stop=True
            break
            
        if stop : 
            break

        if torch.isnan(loss).item() == True : 
            print("nan")
            break
        
        if i%30==0 : 
            print("iteration : ", i, " ; mses : ", loss.item())
            print("diff : ", diff)
            print("Gradient Norm : ", grad_norm.item())
        
        if stop : 
            break
        

    return mses, grad_list, a_list, b_list
    
def score_parra(optim_init, tree_learn, tree_test,alpha,beta,pen_l,pen_m,ntimes,statistic="LTT"): 
    assert statistic in ["likelihood","LTT"]
    
    a_i1=optim_init.a_init.clone().detach().requires_grad_(True)
    b_i1=optim_init.b_init.clone().detach().requires_grad_(True)
    pen=penalties(pen_l,pen_m,alpha,beta)
    score=0

    for k in range(len(tree_learn)):
        print("tree leanrn n : ", k)
        
        optim0=estim_param(a_i1,b_i1,optim_init.n_iter, optim_init.lr0,optim_init.opt)
        res=optim_multiple_trees(optim0, tree_learn[k], pen,statistic)
        
        a_est=res[2][-1].clone().detach()
        b_est=res[3][-1].clone().detach()
        
        score += (1/len(tree_learn))*objective(torch.exp(a_est),torch.exp(b_est),tree_test[k],penalties(pen.pen_l,pen.pen_m,0,0),statistic)                    
            
    return score,alpha,beta
             

def score(optim_init, tree_learn, tree_test,alpha_grid,beta_grid,pen_l,pen_m,ntimes,statistic="LTT"): 
    assert statistic in ["likelihood","LTT"]
    #on calcule l'estimation sur chaque sous ensemble d'apprentissage en partant du précédent : 
    n=len(alpha_grid)
    Ma=torch.tensor(np.zeros((n,n,ntimes)))
    Mb=torch.tensor(np.zeros((n,n,ntimes)))


    Ma[0,0]= optim_init.a_init
    Mb[0,0]= optim_init.b_init
    M_score=np.zeros((n,n))
    
    for i in range(n): 
        for j in range(n):

            pen=penalties(pen_l,pen_m,alpha_grid[i],beta_grid[j])
            a_i1=Ma[max(i-1,0),j]
            b_i1=Mb[max(i-1,0),j]

            optim0=estim_param(a_i1,b_i1,optim_init.n_iter, optim_init.lr0,optim_init.opt)
            res=optim_multiple_trees(optim0, tree_learn, pen,statistic)
            a_est=res[2][-1].clone().detach()
            b_est=res[3][-1].clone().detach()
                
            score = objective(torch.exp(a_est),torch.exp(b_est),tree_test,penalties(pen.pen_l,pen.pen_m,0,0),statistic)                    
            
            #Pour gagner du temps on ne fait pas ça :

            """
            a_i2=Ma[i,max(j-1,0)]
            b_i2=Mb[i,max(j-1,0)]
            optim0bis=estim_param(a_i1,b_i1,optim_init.n_iter, optim_init.lr0,optim_init.opt)
            res2=optim_multiple_trees(optim0bis, tree_learn, pen,statistic)
            a_est2=res2[2][-1].clone().detach()
            b_est2=res2[3][-1].clone().detach()
            score2 = objective(torch.exp(a_est2),torch.exp(b_est2),f_t,T_t,T_b_test,"stem",T_b[-1])                    
       
            if score < score2 : 
                M_score[i,j] = score.item()
                Ma[i,j]=a_est.detach()
                Mb[i,j]=b_est.detach()    
            else : 
                M_score[i,j] = score2.item()
                Ma[i,j]=a_est2.detach()
                Mb[i,j]=b_est2.detach() 
            """

            M_score[i,j] = score.item()
            Ma[i,j]=a_est.detach()
            Mb[i,j]=b_est.detach() 
            
            print("i : ", i, "j : ", j)
            print("score : ", M_score[i,j])

    return M_score, Ma, Mb
            

#function used to parralelise the loop on B 

def inner_optimization(k,tree_learn,tree_test, Time,theta, lambda_est, mu_est, grad_obj, H,pen_l,pen_m,statistic, n_iter, lr0=0.01, s=0.01):
    a_init = torch.log(lambda_est.clone().detach()) 
    b_init = torch.log(mu_est.clone().detach())

    optim_init = estim_param(a_init, b_init, n_iter, lr0)
    res = optim_multiple_trees(optim_init, tree_learn[k], penalties(pen_l, pen_m, torch.exp(theta[0].detach()), torch.exp(theta[1].detach())), statistic)

    lamb_est_new = torch.exp(res[2][-1]).clone().detach().requires_grad_(True)
    mu_est_new = torch.exp(res[3][-1]).clone().detach().requires_grad_(True)



    z1 = pen(pen_l, lamb_est_new, Time, s)
    z2 = pen(pen_m, mu_est_new, Time, s)
    z1.backward()
    z2.backward()
    grad_P = torch.stack([torch.cat([lamb_est_new.grad.clone().detach(), torch.zeros(len(Time))]),
                         torch.cat([mu_est_new.grad.clone().detach(), torch.zeros(len(Time))])]).float()

    lamb_est_new.grad.zero_() 
    mu_est_new.grad.zero_()

    sk = torch.cat([lamb_est_new - lambda_est, mu_est_new - mu_est]).float()

    z = loss_multiple_trees(lamb_est_new, mu_est_new, tree_learn[k], penalties(pen_l, pen_m, torch.exp(theta[0].detach()), torch.exp(theta[1].detach())), statistic)
    z.backward()
    grad_obj_new=torch.cat([lamb_est_new.grad, mu_est_new.grad])

    lamb_est_new.grad.zero_()
    mu_est_new.grad.zero_()

    yk = (grad_obj_new - grad_obj).float()

    H = H + ((sk.T @ yk + yk.T @ H @ yk) * (sk.reshape(len(sk), 1) @ sk.reshape(1, -1))) / (sk.T @ yk) ** 2 - (
                H * (yk @ sk.T) + (sk @ yk.T) * H) / (sk.T @ yk)

    grad_theta_hat_list = (-H @ grad_P.T)

    y = objective(lamb_est_new, mu_est_new, tree_test[k], penalties(pen_l, pen_m, 0, 0), statistic)
    y.backward()
    J_prime_list = torch.cat([lamb_est_new.grad, mu_est_new.grad]).float()
    lamb_est_new.grad.zero_()
    mu_est_new.grad.zero_()
    
    return grad_theta_hat_list.clone().detach(), J_prime_list.clone().detach(), H.clone().detach(), grad_obj_new.clone().detach(), lamb_est_new.clone().detach(), mu_est_new.clone().detach()

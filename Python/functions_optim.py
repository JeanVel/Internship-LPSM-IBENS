import torch
import numpy as np 
from scipy import integrate
from scipy import interpolate   
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

torch.set_printoptions(precision=10)

### Class definitions ### 


class estim_param:
    def __init__(self,a_init,b_init,n_iter,lr0,opt='adam'):
        self.a_init=a_init
        self.b_init=b_init
        self.n_iter=n_iter
        self.lr0=lr0
        self.opt=opt    


class penalties:
    def __init__(self,pen_l,pen_m,alpha,beta,s=0.01):
        self.pen_l=pen_l
        self.pen_m=pen_m
        self.alpha=alpha
        self.beta=beta
        self.s=s


class tree:
    def compute_LTT(T_b,T): #retrieve LTT from branching times
        LTT=torch.ones(len(T))
        for i in range(len(T)) :
            LTT[i]+=torch.sum(T_b >= T[i]) 
        return LTT
    
    def __init__(self,T_b,T,f,cond="crown",t1=None):
        self.T_branch=T_b
        self.T=T
        self.LTT=tree.compute_LTT(T_b,T)
        self.f=f
        self.cond=cond
        self.t1=t1
    

#differentiable approximation of absolute value 
        
def approx(x,s): 
    return x*torch.erf(x/s)


#Solve EDO with euler scheme (faster than computing the exact solution witht the integrals)

def compute_E(l,m,T,f): 
    E=torch.zeros(len(T))
    E[0]=1-f
    for i in range(1,len(T)):
        lamb=l[i]
        mu=m[i]
        hi=T[i]-T[i-1]
        E[i]=E[i-1].clone()+hi*(mu-E[i-1].clone()*(lamb+mu)+ lamb*E[i-1].clone()**2)
    return E


# Compute dLTT in point t
def M(l,m,t,T,f,M0):
    t_index=torch.searchsorted(T,t)
    x=T[:t_index+1]
    E=torch.zeros(t_index+1)
    E[0]=1-f
    for i in range(1,t_index+1):
        lamb=l[i]
        mu=m[i]
        hi=T[i]-T[i-1]
        E[i]=E[i-1].clone()+hi*(mu-E[i-1].clone()*(lamb+mu)+ lamb*E[i-1].clone()**2)
    y=l[:t_index+1]*(E-1)
    return M0*torch.exp(torch.trapezoid(y,x))

# Compute dLTT for every time points 
def model(l,m,T,f,M0):
    return torch.tensor([M(l,m,T[i],T,f,M0) for i in range(len(T))])

#Mean square error 
def MSE(l,m,tree):
    y=0
    for i in range(len(tree.T)):
        y += torch.square( (tree.LTT[i] - M(l,m,tree.T[i],tree.T,tree.f,tree.LTT[0])))
    return y


### Penalties ###


def L1_exp_x(l,s,T,index) :
    h=T[1]-T[0]
    if index == 0: 
        return approx((((l[index+2]-2*l[index+1]+l[index])*l[index]/(h**2) - ((1/h)*(l[index+1]-l[index]))**2 )/(l[index]**2 ) ),s)
    if index == len(T)-1 :
        return approx((((l[index]-2*l[index-1]+l[index-2])*l[index]/(h**2) - ((1/h)*(l[index]-l[index-1]))**2 )/((l[index])**2 ) ),s)
    else :  
        return approx( ( ((l[index-1]-2*l[index]+l[index+1])*l[index]/(h**2) - ((1/h)* (l[index+1]-l[index]) )**2 )/(l[index])**2 ),s)
    

def L2_exp_x(l,T,index) : #q(theta) 
    h=T[1]-T[0]
    if index == 0 : 
        return (((l[index+2]-2*l[index+1]+l[index])*l[index]/(h**2) - ((1/h)*(l[index+1]-l[index]))**2 )/(l[index]**2 ) )**2
    if index == len(T)-1 :
        return (((l[index]-2*l[index-1]+l[index-2])*l[index]/(h**2) - ((1/h)*(l[index]-l[index-1]))**2 )/((l[index])**2 ) )**2
    else : 
        return( ( (l[index-1]-2*l[index]+l[index+1])*l[index]/(h**2) - ((1/h)* (l[index+1]-l[index]) )**2 )/(l[index])**2  )**2



def L2_sec_x(l,T,index) : #L2 norm second derivative
    h=T[1]-T[0]
    if index == 0 : 
        return ( (2*l[index]-5*l[index+1]+4*l[index+2]-l[index+3]) / (h**3) ) **2
    if index == len(T)-1 :
        return ((2*l[index]-5*l[index-1]+4*l[index-2]-l[index-3]) / (h**3) )**2
    else : 
        return ((l[index+1]-2*l[index]+l[index-1])/ (h**2) )**2


def L1_sec_x(l,s,T,index): #L1 on second derivative
    h=T[1]-T[0]
    if index == 0: 
        return approx( ((2*l[index]-5*l[index+1]+4*l[index+2]-l[index+3])/(h**3)),s)
    if index == len(T)-1 :
        return approx( ((2*l[index]-5*l[index-1]+4*l[index-2]-l[index-3]) /(h**3)),s)
    else :  
        return approx( ((l[index+1]-2*l[index]+l[index-1])/(h**2)),s)
    


def L2_prime_x(l,T,index) : #L2 norm of the first derivative
    h=T[1]-T[0]
    if index ==0 : 
        return (0.5*(-3*l[index] +4*l[index+1] -l[index+2])/(l[index]*h))**2
    
    if index == len(T)-1 :
        return (0.5*(3*l[index] -4*l[index-1] +l[index-2])/(l[index]*h))**2
    else :
        return ((l[index+1]-l[index])/(l[index]*h))**2
    



def L1_prime_x(l,s,T,index): #L1 on first derivative
    h=T[1]-T[0]
    if index == 0: 
        return approx(( (l[index]-l[index+1])/(h*l[index])),s)
    if index == len(T)-1 :
        return approx( ( (l[index]-l[index-1])/(h*l[index])),s)
    else :  
        return approx(((l[index+1]-l[index])/(h*l[index])),s)
    



def L2_exp(l,T): 
    res=0
    for i in range(len(T)):
        res+= L2_exp_x(l,T,i)
    return res

def L1_exp(l,T,s):
    res=0
    for i in range(len(T)):
        res+= L1_exp_x(l,s,T,i)
    return res


def L2_sec(l,T): #penalty L2 on the second derivatives 
    res=0
    for i in range(len(T)): 
        res += L2_sec_x(l,T,i)
    return res

def L2_prime(l,T): #penalty L2 on the first derivative 
    res=0 
    for i in range(len(T)):
        res += L1_prime_x(l,T,i)
    return res


def L1_prime(m,T,s=0.01): #penalize the L1 norm of the first derivative of m
    res=0
    for i in range(len(T)):
        res += L1_prime_x(m,s,T,i)
    return res


def L1_sec(m,T,s=0.01): #penalize the L1 norm of the second derivative of m
    res=0
    for i in range(len(T)):
        res += L1_sec_x(m,s,T,i)

    return res
    

def pen(pen_l,l, T,s=0.01) : #pen = ["form",j] :  form in ["lin", "exp", "cpm"] on j-th derivative of l
    assert pen_l[0] in ["lin","exp","cpm"]
    assert pen_l[1] in [1,2]

    if pen_l[0]=='lin': 
        if pen_l[1]==1 :
            return L1_sec(l,T,s)
        elif pen_l[1] == 2 :
            return L2_sec(l,T)    
    elif pen_l[0]=='exp':
        if pen_l[1]==1 :
            return L1_exp(l,T,s)
        elif pen_l[1] ==2 :
            return L2_exp(l,T)
    elif pen_l[0]=='cpm': 
        if pen_l[1]== 1 :
            return L1_prime(l,T,s)
        elif pen_l[1] == 2 :
            return L2_prime(l,T)



#Regularization term : 
def regu(l,m,T,penalties) :
    return (penalties.alpha)*pen(penalties.pen_l, l, T,penalties.s) + (penalties.beta)*pen(penalties.pen_m, m,T,penalties.s) 
    

### functions used to compute Node depth density

def R(l,m,s,T): #R(s) #s is time index  T
    d=l-m #lambda-mu : tensor of size n
    y=d[s:] #integrand : tensor of size index
    x=T[s:] #time points where d is sampled : tensor of size index
    return torch.trapezoid(y,x)



def F(l,m,t,f,T): # t is a time point of T
    inf_index=torch.searchsorted(T, t)  
    y=0 #initialization
    for i in range(inf_index+1,len(T)):
        y += ( (T[i]-T[i-1]) * 0.5 * ( torch.exp(R(l,m,i,T))*l[i] + torch.exp(R(l,m,i-1,T))*l[i-1]) ) #trapezoid method
    return 1+f*y


#variance of the LTT at time t = t_max-s
def var(l, m, s,T, f ) :   
    p = F(l,m,s,f,T)/F(l,m,0,f,T)
    return ((1-p)/(p**2))
    


### Likelihood computation ###
""""
###conditioning to use to fit the model:

tot_time = the index of age of the phylogeny (crown age, or stem age if known). 

FALSE: no conditioning (not recommended);

"stem": conditioning on the survival of the stem lineage (use when the stem age is known, 
in this case t1 should be the stem age);

"crown" (default): conditioning on a speciation event at the crown age and survival 
of the 2 daugther lineages (use when the stem age is not known, in this case t2 should be the crown age).
"""

def fast_exponentiation(base, exponent):
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base
        base *= base
        exponent //= 2
    return result


#Compute the integral of lambda-mu between 0 and T[s] (s is a time point index)
def integr(l,m,s,T): #s is a time point index 
    d=l-m #lambda-mu : tensor of size n
    y=d[:s+1] #integrand : tensor of size index
    x=T[:s+1] #time points where d is sampled : tensor of size index
    return torch.trapezoid(y,x)

#Compute the doubles integrals using trapezoid method
def double_integr(l,m,t,T): #t is a branching time index
    y=0 #initialization
    for i in range(1,t+1):
        y += ( (T[i]-T[i-1]) * 0.5 * ( torch.exp(integr(l,m,i,T))*l[i]+torch.exp(integr(l,m,i-1,T))*l[i-1]) ) #trapezoid method
    return y

def psi_hat(l,m,t,f,T): #t is a branching time index
    #index=torch.searchsorted(T,t)
    fact = torch.exp(integr(l,m,t,T))
    num = double_integr(l,m,t,T)
    return fact*(1+f*num)**(-2)

def phi(l,m,t,f,T): #t is a branching time index
    num=torch.exp(integr(l,m,t,T))
    denum=1/f+double_integr(l,m,t,T)
    return 1-num/denum

def loglikelihood(l,m,tr):
    n_tips=len(tr.T_branch)+1
    loglikelihood=0
    for j in range(1,n_tips):
        index_ti=torch.searchsorted(tr.T, tr.T_branch[-j]) #index of j-th branching time in T
        psi=psi_hat(l,m,index_ti,tr.f,tr.T)
        loglikelihood += torch.log(psi) + torch.log(l[index_ti])
    

    tot_time_index=torch.searchsorted(tr.T, tr.T_branch[-1])
    loglikelihood+=torch.log(psi_hat(l,m,tot_time_index,tr.f,tr.T))
    loglikelihood+=n_tips*torch.log(torch.tensor(tr.f))

    if tr.cond == False :
        return loglikelihood
    elif tr.cond=="stem" :
        index_t1=torch.searchsorted(tr.T, tr.t1)
        return loglikelihood-torch.log(1-phi(l,m,index_t1,tr.f,tr.T) ) 
    elif tr.cond=="crown" :
        return loglikelihood - torch.log(l[tot_time_index]) - 2*torch.log(1-phi(l,m,tot_time_index,tr.f,tr.T) )



#objective function with penalties :
def objective(l,m,tree,penalties,statistic): 
    if statistic=="likelihood":
        return ( -loglikelihood(l,m,tree) + (1/len(tree.T)) * regu(l,m,tree.T,penalties) )
    elif statistic=="LTT":
        return 1/len(tree.T)*( MSE(l,m,tree)+regu(l,m,tree.T,penalties))



#Gradient descent : 
def GD(estim_param, tree, penalties, statistic="LTT",eps=0.001, threshold=0.01, patience=3): 
    assert statistic in ["likelihood","LTT"]
    mses=[] #to store the evolution of the objective function during the gradient descent
    a=estim_param.a_init.clone().detach().requires_grad_(True)
    b=estim_param.b_init.clone().detach().requires_grad_(True)
    grad_list=[]
    a_list=[a.clone().detach()]
    b_list=[b.clone().detach()]
    stop=False
    count=0
    epoch=0

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

        loss=objective(torch.exp(a),torch.exp(b),tree,penalties,statistic)
        optimizer.zero_grad()
        loss.backward()
        grad_norm=torch.norm(torch.cat([a.grad,b.grad]))
        grad_list.append(grad_norm.item())
        optimizer.step()
        lr_scheduler.step(loss)


        a_list.append(a.clone().detach())
        b_list.append(b.clone().detach())

        mses.append(loss.item())




        #stopping condition : 
        diff=torch.norm(torch.cat([a_list[-1]-a_list[-2], b_list[-1]-b_list[-2] ]))
        if diff < eps :
            count +=1
            #print("diff : ", diff)
            #print("count : ", count)
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
        
        #if i%30==0 : 
        #print("diff : ", diff)
        #print("iteration : ", i, " ; mses : ", loss.item())
        #print("Gradient Norm : ", grad_norm.item())
        
        if stop : 
            break
        

    return mses, grad_list, a_list, b_list


def plot_estim(tree,res,scale="log",init=False,real=None) :
    lamb_est=torch.exp(res[2][-1])
    mu_est=torch.exp(res[3][-1])
    
    plt.plot(tree.T,lamb_est,label="lambda")
    if init :
        plt.plot(tree.T,torch.exp(res[2][0]),label="lambda_init")
    if scale=="log":
        plt.xscale('log')
        plt.xlabel("log time")

    plt.xlabel("time")
    plt.ylabel("lambda")
    plt.legend()
    plt.show()
    
    plt.plot(tree.T,mu_est,label="mu")
    if init :
        plt.plot(tree.T,torch.exp(res[3][0]),label="mu_init")
    if scale=="log":
        plt.xscale('log')
        plt.xlabel("log time")
    
    plt.xlabel("time")
    plt.ylabel("mu")
    plt.legend()
    plt.show()

    plt.plot(tree.T,tree.LTT,label="LTT")
    plt.plot(tree.T,model(lamb_est,mu_est,tree.T,tree.f,tree.LTT[0]),label="model")
    plt.xlabel("time")
    plt.ylabel("LTT")
    plt.legend()
    plt.show()

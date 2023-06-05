import numpy as np
import pandas as pd



@jit
def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi



@jit

def MIcheck(inp, tau, dim):
    scor=[]


    for tau in range(1,tau):
        sd2=[]
        sd2=inp[:-dim*tau]

        for d in range(1,dim):
            sd2=np.vstack((sd2,inp[tau*d:-((dim-d)*tau)])) #stack the next dimension
            sd3=np.transpose(sd2)

            
            araydelta=space.distance_matrix(sd3[0:1], sd3)
            num_delta=araydelta[araydelta<=delta]

            sd3=inp[tau*d:-((dim-d)*tau)]

            sd3=sd3.reshape(-1,1)
            araydelta=space.distance_matrix(sd3[:1], sd3)
            num_eps=len(araydelta[araydelta<=delta])



            scor.append((tau,d,num_eps/num_delta))


    return scor






def embedding(vect, n_dim, lag):


    embd=np.zeros((n_dim, len(vect)-(n_dim*lag)))
    for i in range(1, n_dim):
        embd[i]=(np.array(vect[lag*i:-((n_dim-i)*lag)]))
        
    s,v,d=np.linalg.svd(embd)

    return v, len(v)




@jit
def euc_dist(x1, x2):
    dist=np.sum(np.sqrt(np.square(x2-x1)))
    return  dist


@jit
def wolf_lyp(inp, lypsamp, lyp_tr, delta_t): #inp should be a numpy array in shape (obs, dim), 
    

    leng=len(inp)
    dist=np.empty(leng/2)
    i=0
    for elem in inp[:leng/2]:
        dist[i]=np.sum(np.sqrt(np.square(inp-elem)))
        i+=1
    
    nearn1, nearn2=np.sort(dist)[1:2]
    nn_ind_1=np.where(dist[dist==nearn1])
    nn_ind_2=np.where(dist[dist==nearn2])

    samps=np.sort(np.random.randint(0, leng/2, size=lyp_tr))

    lyp=np.empty(samps)


    i=0
    for j in range(len(samps)):
        

        x1=inp[nn_ind_1+samps[j]]
        x2=inp[nn_ind_2+samps[j]]

        rate=euc_dist(x1, x2)/(delta_t*(nn_ind_1+samps[j-1]-samps[j]))
        lyp[j]=np.log(rate)

        i+=1


    return lyp



'''
CCM OLD Codes

'''

#%%

import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr
import embedding
import dlc_tda_ppross as dlcp
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import jit, njit, prange



#%%
import numpy as np
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
import embedding
import matplotlib.pyplot as plt
from numba import jit



@jit(nopython=True)
def find_ind(inp, l):
    j=0
    for i in inp:
        if i==l:
            return float(j)
        else:
            j+=1


@jit(nopython=True)
def euc_dist(p1, v2): #p1: point, v1:vector

    dist_pt=np.zeros(len(p1))
    dist_vct=np.zeros(len(v2))
    k=0
    for j in v2:
        for i in range(len(p1)):
            dist_pt[i]=(p1[i]-j[i])**2
        dist_vct[k]=np.sqrt(np.sum(dist_pt))
        k+=1

    return dist_vct

@jit(nopython=True)
def simp_nnb(inp,t_point):
    dist1=euc_dist(inp[t_point], inp)
    dist2=np.sort(dist1)

    near_n_ind=np.zeros_like(dist2[1:])
    k=0
    for dis in dist2[1:]:    
        near_n_ind[k]=find_ind(dist1, dis)
        k+=1
    return   near_n_ind, dist2[1:]


@jit(nopython=True)
def euc_dist1(p1,p2): #p1 and p2 are n-dimentiona points
    dist_pt=np.zeros(len(p1))
    for i in range(len(p1)):
        dist_pt[i]=(p1[i]-p2[i])**2
    return np.sqrt(np.sum(dist_pt))



#scoring the predictions

def score(preds, actual):
    """The coefficient R^2 is defined as (1 - u/v), where u is the regression
    sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
    sum of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible
    score is 1.0, lower values are worse.
    Parameters
    ----------
    preds : 1d array
        Predicted values.
    actual : 1d array
        Actual values from the testing set.
    Returns
    -------
    cc : float
        Returns the coefficient of determiniation between preds and actual.
    """

    u = np.square(actual - preds ).sum()
    v = np.square(actual - actual.mean()).sum()
    r2 = 1 - u/v

    return r2



def CCM(inp, trgt, dim, forecast_l,train_frac):

    inp_t_p=int(inp.shape[0]*train_frac)
    inp_t=inp[:inp_t_p]

    #calculate the weights
    ind_in ,dist=simp_nnb(inp_t,0)

    wi=np.zeros(dim+1)
    k=0
    for i in ind_in[:dim+1]:
        wi[k]=np.exp(-dist[i.astype(np.int32)]/dist[0])
        k+=1
    s_wi=np.sum(wi)





    #make predictions 
    prediction=np.zeros((forecast_l,inp.shape[1]))
    m=0
    for i in range(forecast_l):
        i=i+inp_t_p #for prediction after the training fraction
        ind_in ,_=simp_nnb(inp,i)
        y_hat=np.array([0.0]*inp.shape[1])
        for j in range(dim+1):
            y_hat+=((wi[j]*trgt[ind_in[j].astype(np.int32)])/s_wi)
        prediction[m]=y_hat
        m+=1




    #score predictions 
    scr=np.zeros(forecast_l-5)
    m=0
    for p in range(5,prediction.shape[0]):
        scr[m]=score(inp[inp_t_p:inp_t_p+p], prediction[:p])
        m=m+1


    return scr, prediction


#%% Data


            
case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/SHM40.csv'



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)

embeded=embedding.l_embed(d_body, 100, 1)
embeded2=np.swapaxes(embeded,0,1)
u,s, v=np.linalg.svd(embeded2)
emb=np.swapaxes(u[:5], 0,1)

            
case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LID23.csv'



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)

embeded=embedding.l_embed(d_body, 100, 1)
embeded2=np.swapaxes(embeded,0,1)
u,s, v=np.linalg.svd(embeded2)
emb2=np.swapaxes(u[:5], 0,1)






inp2=emb
inp1=emb2
leng1=len(inp1)
leng2=len(inp2)
#n=np.array([distance.euclidean(inp1[0],temp) for temp in inp1[:leng1]])

dim=15 #equal to embedding dimentions




#%%
man_1=inp1
man_2=inp2
dim=15

#%%
@jit(nopython=True)
def ccm(man_1,man_2,dim,n_samples):
    initial_ind=np.random.randint(0,len(man_1),n_samples)
    for index in initial_ind:
        dists, nn_list=ccm_nnb(man_1, index)  #correct for digit time point

        u=dists/dists[0]
        sum_u=np.sum(u)
        w=(u/sum_u).reshape(-1,1)

    scor=np.zeros(len(man_1)-1-dim+1)
    j=0
    for i in range(dim+1,len(man_1)):

        tmp=np.empty_like(man_1[:i])
        k=0
        for l in nn_list[:i]:
            tmp[k]=man_1[int(l)]
            k=k+1

        y_mx=np.sum(tmp*w[:i],axis=0)
        #y_mx=np.sum(man_1.take(nn_list[:i].astype(int), axis=0)*w[:i],axis=0)




        scor[j]=np.corrcoef(y_mx, man_2[0])[0][0]
        
        j=j+1
    return scor

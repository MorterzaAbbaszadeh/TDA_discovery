import embedding as emb 
import os
import numpy as np
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
from numba import jit, njit, prange
import pandas as pd




'''
LIBRARY
'''




@jit(nopython=True)
def find_ind(inp, l):

    '''
    Finds index of a given element in an array
    ----
        inp: the vector to search in
        l: the target we want to find the index for
    ---
        j: is a float, for JIT maintenence
    '''
    
    
    j=0
    for i in inp:               #condition. l=inp[j]
        if i==l:                #if the condition is true report the counter
            return float(j)
        else:                   #if not cycle
            j+=1





@jit(nopython=True)
def euc_dist(p1, v2): #p1: point, v1:vector
    
    '''
        the euclidian distance between a point and individual elements of a vector, 
        Both should have the same dimentions
    ----
        p1:
        v2: 
    '''


    dist_pt=np.zeros(len(p1))                    #initialization
    dist_vct=np.zeros(len(v2))
    k=0


    for j in v2:
        for i in range(len(p1)):                  #calculate distance along each individual dimention (length of the point p)
            dist_pt[i]=(p1[i]-j[i])**2 
        dist_vct[k]=np.sqrt(np.sum(dist_pt))     #sum up the distances and take the square root
        k+=1

    return dist_vct



@jit(nopython=True)
def simp_nnb(inp,t_point):

    '''
    reports the indices and distances of the nearest neighb. of t_point (index of an element in inp)
    ---
        inp: manifold
        t_poin: int, the index of the time point for near neighb
    ---
        near_n_ind: indicies of the nearest neighbours
        dist: distances with the nearest neighbours
    '''

    dist1=euc_dist(inp[t_point], inp)       #get the distance vector between inp vector and inp[t_point] point
    dist2=np.sort(dist1)                    #sort the distance vector to get the neighbourhood


    near_n_ind=np.zeros_like(dist2[1:])
    k=0


    for dis in dist2[1:]:    #find the index of the sorted distances in the distance vector to get the indicies of the near neighb.
        near_n_ind[k]=find_ind(dist1, dis) 
        k+=1



    return   near_n_ind, dist2[1:] #dont report the distance with the point itself (0)




@jit(nopython=True)
def euc_dist1(p1,p2): #p1 and p2 are n-dimentiona points


    '''
        Same as the euclidian distance between point and vector but for two points
        python shnanigans
    ----
        p1, p2: two points in the same dimensions.

    '''


    dist_pt=np.zeros(len(p1))
    for i in range(len(p1)):
        dist_pt[i]=(p1[i]-p2[i])**2




    return np.sqrt(np.sum(dist_pt))



#scoring the predictions
@jit(nopython=True)
def score(preds, actual):

    'This part of the code for scoring the predictions was taken directly from CCM package'


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


#score for each point of the prediction
@jit(nopython=True)
def increamental_simp_score(prediction, inp, forecast_l, track_p):

    #score predictions 
    scr=np.zeros(forecast_l-1)
    m=0
    for p in range(1,prediction.shape[0]):
        scr[m]=score(prediction[:p],inp[track_p:track_p+p])
        m=m+1
    return scr


#score for each point of the prediction
@jit(nopython=True)
def mean_simp_score(prediction, inp, forecast_l, track_p):

    #score predictions 
    scr=np.zeros(forecast_l-1)
    m=0
    for p in range(1,prediction.shape[0]):
        scr[m]=score(prediction[:p],inp[track_p:track_p+p])


    scor=np.mean(scr)
    return scor



@jit(nopython=True)
def simplex_pr(inp,dim, forecast_l,train_p, trk_p):

    '''
    Forcasts the manifold
    -----
        inp: the manifold
        dim: number of the dimentions in the manifold
        forecast_l: number of time points to be forecasted
        train_frac: the fraction of time 
    -----
        scr: scores of the predicted vector, scr[i] corresponds to score of predicion [:i]
        prediction: the prediction manifold
    '''


    #### Data prep
    inp_t_p=train_p
    track_p=inp_t_p-trk_p #this value is very important higher the number, higher the forcast 

    inp_t=inp[:inp_t_p]

    #calculate the weights
    ind_in ,dist=simp_nnb(inp_t,track_p) #0 to inp_t_p

    wi=np.zeros(dim+1)+0.001
    k=0
    for i in ind_in[:dim+1].astype(np.int64):
        wi[k]=np.exp(-dist[i-1]/dist[0]) #-1 to correct for python indexing
        k+=1






    #make predictions 
    prediction=np.zeros((forecast_l,inp.shape[1]))+500  #to avoid division for zero and give a nice spread

    m=0
    for i in range(forecast_l):

        
        y_hat=np.array([0.0]*inp.shape[1])
        s_wi=0
        
        for j in range(dim+1):
            try:
                y_hat+=wi[j]*inp_t[int(ind_in[j])+i]
                s_wi+=wi[j]
            except:
                pass
        if s_wi>0:
            prediction[m]=y_hat/s_wi
        m+=1
    
    scr=increamental_simp_score(prediction, inp, forecast_l, track_p)

    return scr,prediction




'''

Work Functions

'''





def prediction_scores(fname, manif, dim, samps,forecast_l, trk_p):  #should: file name, manif, inputs to simpx, replaced training fraction with 
    outpt=[]

    
    for di in range(2, dim):
        new_man=manif.transpose()[:di]
        new_man=new_man.transpose()
        

        for ind in range(len(samps)):
            try:
                scr,_=simplex_pr(new_man, di, forecast_l, samps[ind], trk_p)  #simplex_pr(inp,dim, forecast_l,train_p, trk_p):
                per_len=len(scr[scr>0.85])
                myn=np.mean(scr[:per_len])
                outpt.append((fname[:3], fname[3:5],samps[ind], di, per_len, myn))
            except: 
                pass #equal to null 


@jit
def inputing(case):

    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')
    
    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)


    #embed and get lags by SVD
    embeded=emb.l_embed(d_body, 100, 1)
    u=emb.svd_lags(embeded,100)

    return u



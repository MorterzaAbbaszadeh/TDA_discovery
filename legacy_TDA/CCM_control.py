



import numpy as np
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
import embedding
import matplotlib.pyplot as plt
from numba import jit
import os
import pandas as pd
import random

def get_body(case, dim):
    



    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)



    #dim=25
    #Shuffling
    #d_body=np.random.shuffle(d_body)
    #np.random.shuffle(d_body)



    #SVD Embedding
    embeded=embedding.l_embed(d_body,100,1)
    manif=embedding.svd_lags(embeded, dim)

    return manif

def get_head(case, dim):


    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2

    cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
     m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
     dlcp.retrivedata(case,'headL.1'))


    embeded=embedding.l_embed(cos_head_ang,100,1)
    manif=embedding.svd_lags(embeded, dim)



    return manif




def get_manif(case, dim):
    



    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)



    #dim=25
    #Shuffling
    #d_body=np.random.shuffle(d_body)
    #np.random.shuffle(d_body)



    #SVD Embedding
    embeded=embedding.l_embed(d_body,100,1)
    manif=embedding.svd_lags(embeded, dim)

    return manif

def get_target(case, dim):


    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2

    cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
     m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
     dlcp.retrivedata(case,'headL.1'))


    embeded=embedding.l_embed(cos_head_ang,100,1)
    manif=embedding.svd_lags(embeded, dim)
    return manif




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



@jit(nopython=True)
def scoring(preds, actual):
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

    u = np.square(actual - preds).sum()
    v = np.square(actual - actual.mean()).sum()
    r2 = 1 - u/v

    return r2





@jit(nopython=True)
def CCM(cause, targt, forecast_l):

    t_points=np.arange(forecast_l) #should creat a list of time points.
    dim=cause.shape[1]
    recons=np.zeros((forecast_l,cause.shape[1]))    
    m=0

    for t_point in t_points:


        #calculate the weights
        ind_in ,dist=simp_nnb(cause,t_point)

        wi=np.zeros(dim+1)
        k=0
        for i in ind_in[:dim+1].astype(np.int64):
            wi[k]=np.exp(-dist[i-1]/dist[0])
            k+=1
        s_wi=np.sum(wi)



        #make predictions 
    

        #i=i+inp_t_p #for prediction after the training fraction
        #ind_in ,_=simp_nnb(targt,t_point)

        y_hat=np.array([0.0]*targt.shape[1])
        for j in range(dim+1):
            y_hat+=((wi[j]*targt[int(ind_in[j])])/s_wi)
        recons[t_point]=y_hat
        m=m+1


    
    scr=scoring(recons, targt[:forecast_l,:])
    
    return recons, scr


@jit(nopython=True)
def get_ccms(cause, targt, dims, forecast_l):

    ccm_scores=np.zeros(dims-2)+50 #arbitary above full range value to detect failure
    m=0
    for i in range(2, dims):
        _, scr=CCM(cause[:,:i], targt[:,:i], forecast_l)
        ccm_scores[m]=scr
        m=m+1

    return  ccm_scores


#first get a list of csv files

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'
csv_list=[]
fname_list=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.startswith(('SHM', 'LES')) and fname.endswith('csv') :
            case=root+'/'+fname #construct the path to file
            csv_list.append(case)
            fname_list.append(fname)


# define the parameters and run this bad boy
dim=70
dims=dim
forecast_l=300
saving_path='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/simplex_proj/'


#make a shufflled list:
tmp_list=random.sample(csv_list, len(csv_list))
shuf_pairs=list(zip(csv_list, tmp_list))

df_list=[]
for i in range(len(fname_list)):
    
    cause=get_body(shuf_pairs[i][0], dim)[::10] #downsampling is essential
    print(shuf_pairs[i][0])
    
    targt=get_head(shuf_pairs[i][1], dim)[::10]
    print(shuf_pairs[i][1])

    output_scr=get_ccms(cause, targt, dims, forecast_l)

    for j in range(len(output_scr)):
        df_list.append(('Shuffled', output_scr[j], j))  #animal no, treatment, score, dimention.


df=pd.DataFrame(df_list, columns=['treatment', 'score', 'dimention'])
df.to_csv(saving_path+'CCM_control.csv')



'''''


NEW Try


''''
#%% CCM Toolbox




import numpy as np
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
import embedding
import matplotlib.pyplot as plt
from numba import jit

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


def get_d_body(case, dim):
    



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

    return d_body
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
def correlation_scoring(targt, predict, forecast_l):



    corrs=np.zeros(targt.shape[1])
    forc_l=len(predict[0])

    m=0
    for i in range(targt.shape[1]):
        corrs[m]=np.corrcoef(targt[:forecast_l,i],predict[:forecast_l,i])[0,1]
        m=m+1




    return corrs





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


#%% CCM2


dim=7

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LES3.csv'

d_body=get_d_body(case, dim)
cause=get_target(case, dim)[::10] #downsampling is essential

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LES3.csv'
targt=get_manif(case, dim)[::10]

plt.plot(d_body)

#%%
forecast_l=200

prediction, score=CCM(cause, targt, forecast_l)


print(score)



#%%

fig=plt.Figure((2,7))
plt.imshow(targt[:500].transpose()[:10], aspect=24)


#%%

fig=plt.Figure((2,7))
plt.imshow(cause[:500].transpose()[:10], aspect=24)

# %%
survey_dim=0


plt.plot(targt[:forecast_l,survey_dim])
plt.plot(prediction[:forecast_l,survey_dim])




#%% Correlation check

import seaborn as sns 



survey_dim=2
sns.regplot(targt[:forecast_l,survey_dim],prediction[:forecast_l,survey_dim])
np.corrcoef(targt[:forecast_l,survey_dim],prediction[:forecast_l,survey_dim])[0,1]



# %%

survey_dim=0
plt.plot(targt[:,survey_dim])
plt.plot(cause[:,survey_dim])


np.corrcoef(targt[:,survey_dim], cause[:,survey_dim])[0][1]




# %%
score(targt[:forecast_l], prediction)
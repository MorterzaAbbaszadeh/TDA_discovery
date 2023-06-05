
#%%
import numpy as np
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
import embedding
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns


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



def simplex_pr(inp,dim, forecast_l,train_frac, trk_p):

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
    inp_t_p=int(inp.shape[0]*train_frac)
    track_p=inp_t_p-trk_p #this value is very important higher the number, higher the forcast 

    inp_t=inp[:inp_t_p]

    #calculate the weights
    ind_in ,dist=simp_nnb(inp_t,track_p) #0 to inp_t_p

    wi=np.zeros(dim+1)
    k=0
    for i in ind_in[:dim+1]:
        wi[k]=np.exp(-dist[i.astype(np.int32)-1]/dist[0])
        k+=1
    #s_wi=wi.sum()





    #make predictions 


    prediction=np.zeros((forecast_l,inp.shape[1]))

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



    #score predictions 
    scr=np.zeros(forecast_l-1)
    m=0
    for p in range(1,prediction.shape[0]):
        scr[m]=score(inp[track_p:track_p+p], prediction[:p])
        m=m+1



    return scr,prediction




# %%

from sklearn.utils import shuffle

from time  import time
case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/LID21.csv'



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)


dim=10
#Shuffling
d_body=shuffle(d_body, random_state=1)
#np.random.shuffle(d_body)



#SVD Embedding
embeded=embedding.l_embed(d_body,100,1)
manif=embedding.svd_lags(embeded, dim)


#no SVD Embedding
#emb=embedding.l_embed(d_body,dim,1) #temp
#manif=emb.swapaxes(1,0)

t0=time()



trin_frac=0.73
frecast_l=300


#simplex_pr(inp,dim, forecast_l,train_frac, trk_p)
scr,prediction=simplex_pr(manif,dim, frecast_l,trin_frac,5)


t1=time()
print(t1-t0)
plt.plot(scr)






# %%

inp=manif
inp_t_p=int(inp.shape[0]*trin_frac)
inspect_dim=0


plt.plot(manif[inp_t_p-5:inp_t_p-5+frecast_l,inspect_dim])
plt.plot(prediction[:frecast_l,inspect_dim])
plt.xticks([],[])
plt.yticks([],[])
sns.despine(top=True,bottom=True, left=True, right=True)
plt.plot([10,70], [-13,-13], color='black')
plt.text(20,-13.5, '1 Sec.' )



#%% good predict

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SKF17.csv'



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)


dim=7

d_body=dlcp.smooth_diff(ang_bod)


#SVD Embedding
embeded=embedding.l_embed(d_body,100,1)
manif=embedding.svd_lags(embeded, dim)[::2]


#no SVD Embedding
#emb=embedding.l_embed(d_body,dim,1) #temp
#manif=emb.swapaxes(1,0)

t0=time()



trin_frac=0.75
frecast_l=200


#simplex_pr(inp,dim, forecast_l,train_frac, trk_p)
scr,prediction=simplex_pr(manif,dim, frecast_l,trin_frac,5)


t1=time()
print(t1-t0)
plt.plot(scr)

# %%

inp=manif
inp_t_p=int(inp.shape[0]*trin_frac)
inspect_dim=1

plt.plot(manif[inp_t_p-5:inp_t_p-5+frecast_l,inspect_dim])
plt.plot(prediction[:frecast_l,inspect_dim])
plt.xticks([],[])
plt.yticks([],[])
sns.despine(top=True,bottom=True, left=True, right=True)
plt.plot([0,0], [-30,30], color='r')
plt.text(5,-25.5, 'Prediction point', color='r' )



# %%


frecast_l=100

plt.plot(manif[inp_t_p:inp_t_p+frecast_l,inspect_dim], manif[inp_t_p:inp_t_p+frecast_l,inspect_dim+1])
plt.plot(prediction[:frecast_l,inspect_dim], prediction[:frecast_l,inspect_dim+1])


#%%
len(scr[scr>0.9])
# %% pickling the data 
import pickle


#pickle.dump(manif[inp_t_p-5:inp_t_p-5+frecast_l], open( "manif.pkl", "wb" ) )

#pickle.dump(prediction[:frecast_l], open( "prediction.pkl", "wb" ) )
# %%
prediction[:frecast_l]



''''Combined Embedding'''



#%%

#def inputting 

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SHM21.csv'




m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
    m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
    dlcp.retrivedata(case,'headL.1'))



ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)



inp=np.array([cos_head_ang,d_body])


dim=5
#embed and get lags by SVD
embeded=embedding.multi_d_embedding(inp,n_tau=100, tau=1)


#inp=d_body
#embeded=embedding.l_embed(inp,n_tau=100, tau=1)


manif=embedding.svd_lags(embeded,dim)[::2]




#predict

trin_frac=0.55
frecast_l=200

scr,prediction=simplex_pr(manif,dim, frecast_l,trin_frac,1)




#%%

inspect_dim=0
plt.plot(manif[inp_t_p:inp_t_p+frecast_l,inspect_dim])
plt.plot(prediction[:frecast_l,inspect_dim])



# %%

plt.plot(manif[:1500,inspect_dim])

# %%
plt.plot(prediction[:1500,inspect_dim])



'''Arvind Controls''' 


# %%


case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SHM21.csv'




m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
    m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
    dlcp.retrivedata(case,'headL.1'))



ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)


bod_ar=dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)

manif=np.array([head_ang,d_body, bod_ar]).T

dim=2

#predict

trin_frac=0.69
frecast_l=300
inp=manif
inp_t_p=int(inp.shape[0]*trin_frac)

scr,prediction=simplex_pr(manif,dim, frecast_l,trin_frac,20)


#%%

inspect_dim=0

plt.plot(manif[inp_t_p:inp_t_p+frecast_l,inspect_dim])
plt.plot(prediction[:frecast_l,inspect_dim])
plt.plot([20, 20], [-2, 1], color='r')

# %%


plt.plot(manif[:1500,inspect_dim],manif[:1500,inspect_dim+1] )


# %%

dim=2

#predict
trin_frac=0.79
frecast_l=300
inp=manif
inp_t_p=int(inp.shape[0]*trin_frac)



scr,prediction=simplex_pr(manif,dim, frecast_l,trin_frac,20)


#%%

inspect_dim=1

plt.plot(manif[inp_t_p:inp_t_p+frecast_l,inspect_dim])
plt.plot(prediction[:frecast_l,inspect_dim])
plt.plot([20, 20], [-2, 1], color='r')

#%%

# %%

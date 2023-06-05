
#%%

import numpy as np
import pandas as pd
from numba import jit
import dlc_tda_ppross as dlcp
import matplotlib.pyplot as plt
import embedding
import os
import seaborn as sns
from scipy.spatial import distance



def total_lyp(inp, n_nn, m, dim):  #input should be the transpose matrix of U
    #inp=inp.T

    leng=len(inp)
    r=np.zeros((dim,leng))

    #determine the long steps:    
    for i in range(int(leng/m)-1):
        
        
        #Initialize
        x_0=inp[i]
        x_m=inp[i+1]
        




        #Get NN
        dist=np.empty(len(inp))

        dist=np.array([distance.euclidean(x_0,temp) for temp in inp])
        dist2=dist
        near_n_i=np.sort(dist2)[:n_nn]
        near_n_0=np.array([inp[np.where(dist==i)[0][0]] for i in near_n_i])

        dist=np.array([distance.euclidean(x_m,temp) for temp in inp])
        dist2=dist
        near_n_i=np.sort(dist2)[:n_nn]
        near_n_m=np.array([inp[np.where(dist==i)[0][0]] for i in near_n_i])


        #Fit least squares
        T_i=np.linalg.lstsq(near_n_0,near_n_m)[0] #T_i This is the problematic part


        if i==0:
            q_i=np.identity(T_i.shape[0])
        

        trg=T_i*q_i
        q_i,r_i=np.linalg.qr(trg)


        r[:,i]=np.log(np.absolute(np.diagonal(r_i)))


    return r


@jit
def sns_ready(inp, fname):
    Dframe=[]
    tau=0
    for i in inp:
        Dframe.append((np.log(i),tau,fname))
        tau+=1

    return Dframe


@jit
def svd_s(inp):
    u,s, v=np.linalg.svd(inp)
    norm_s=sum(np.sqrt(np.square(s)))
    s_norm=s/norm_s

    return s_norm







#%%

df=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'

for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname.endswith('.csv') and fname[:3] in (['SUM', 'LID', 'SKF']):
            
            case=root+'/'+fname #construct the path to file


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_body=dlcp.smooth_diff(ang_bod)

            #c_ang_bod=dlcp.cos_angs(tail_x, tail_y, m_head_x, m_head_y)
            #d_body=dlcp.smooth_diff(c_ang_bod)

            #d_ar=np.array(dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)) #no significant difference
            #d_body=dlcp.smooth_diff(d_ar)



            #cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                    #m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                #dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))

            #d_body=dlcp.smooth_diff(cos_head_ang)



            embeded=embedding.l_embed(d_body, 100, 1)
            embeded2=np.swapaxes(embeded,0,1)
            embeded2=embeded2-np.mean(embeded2)
            #u,s, v=np.linalg.svd(embeded2)
            #s=s/np.sum(s)

            s=svd_s(embeded2)
            #emb=np.swapaxes(u[:15], 0,1)
            df2=sns_ready(s, fname[:3])
            df=df+df2


dframe=pd.DataFrame(df, columns=['delta','tau','fname']) 





#%%
dframe_t=dframe[dframe['fname'].isin(['LID', 'SKF'])]


sns.lineplot(x='tau', y='delta', hue='fname', data=dframe_t, ci=95)
#plt.xlim(0,2)
#plt.ylim(-3, 0)





#%%
from scipy import stats
dframe_p=dframe_t[dframe_t['tau']==40]
sns.boxplot(x='fname', y='delta',data=dframe_p)



sign=stats.mannwhitneyu(dframe_p[dframe_p['fname']=='SKF']['delta'].values,
                        dframe_p[dframe_p['fname']=='SUM']['delta'].values)
print(sign)












#%%
case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LID28.csv'



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





#%%
#plt.plot(u[:,1],u[:,2])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=u[:2000,2],ys=u[:2000,3],zs=u[:2000,4],alpha=1)

#%%

r=total_lyp(emb, 15, 15, 5)
print(np.mean(r, axis=1))


#%%
u2=np.dot(embeded2, v.T)




#%%
plt.plot(u2[:5000,2],u2[:5000,3], linewidth=0.3,color='black', alpha=0.8)




# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=u2[:7000,0],ys=u2[:7000,1],zs=u2[:7000,2], linewidth=0.3,color='black', alpha=0.8)
# %%

plt.plot(np.log(s))


#%%Simplex toolbox
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



@jit(nopython=True)
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


#%%
'''
NEW PILOT
'''



#%% NEw simplex projection pilot

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LID21.csv'



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)



dim=25
#Shuffling
#d_body=np.random.shuffle(d_body)
#np.random.shuffle(d_body)



#SVD Embedding
embeded=embedding.l_embed(d_body,100,1)
manif=embedding.svd_lags(embeded, dim)


#non-SVD Embedding
#emb=embedding.l_embed(d_body,dim,5) #temp
#manif=emb.swapaxes(1,0)

inp=manif

train_frac=0.68 #lowering reduces accuracy
forecast_l=255





#simp(inp,  dim, forecast_l,train_frac): 



inp_t_p=int(inp.shape[0]*train_frac)
track_p=inp_t_p-25      #this value is very important higher the number, higher the forcast 

inp_t=inp[:inp_t_p]

#calculate the weights
ind_in ,dist=simp_nnb(inp_t,track_p) #0 to inp_t_p

wi=np.zeros(dim+1)
k=0
for i in ind_in[:dim+1]:
    wi[k]=np.exp(-dist[i.astype(np.int32)]/dist[0])
    k+=1


s_wi=wi.sum()



#make predictions 
prediction=np.zeros((forecast_l,inp.shape[1]))

m=0
for i in range(forecast_l):
    #i=i+inp_t_p #for prediction after the training fraction
    #ind_in ,_=simp_nnb(inp,i)
    
    y_hat=np.array([0.0]*inp.shape[1])
    #s_wi=0
    for j in range(len(wi)):
        try:
            y_hat+=wi[j]*inp_t[ind_in[j].astype(np.int32)+i]
            
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



#%%

di=3
new_man=manif.swapaxes(0,1)[:di]
new_man.swapaxes(1,0)

# %%
inspect_dim=7


plt.plot(inp[track_p:track_p+forecast_l,inspect_dim])



plt.plot(prediction[:forecast_l,inspect_dim])



#%%

plt.plot(inp[inp_t_p:inp_t_p+forecast_l,inspect_dim],inp[inp_t_p:inp_t_p+forecast_l,inspect_dim+1])

plt.plot(prediction[:forecast_l,inspect_dim],prediction[:forecast_l,inspect_dim+1])



# %%
plt.plot(scr)
plt.ylim(0,1)

# %%
plt.plot(cor)

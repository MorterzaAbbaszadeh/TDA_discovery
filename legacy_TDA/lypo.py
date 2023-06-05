#%%
import numpy as np
from scipy.spatial import distance
import embedding
import dlc_tda_ppross as dlcp
from numba import jit




#defunct
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
def nnb(inp,t_point):
    dist1=euc_dist(inp[t_point], inp)
    dist2=np.sort(dist1)
    near_n_ind=find_ind(dist1, dist2[1])
    return  dist2[1], near_n_ind



@jit(nopython=True)
def euc_dist1(p1,p2): #p1 and p2 are n-dimentiona points
    dist_pt=np.zeros(len(p1))
    for i in range(len(p1)):
        dist_pt[i]=(p1[i]-p2[i])**2
    return np.sqrt(np.sum(dist_pt))


@jit
def sns_ready(inp, fname ,tau):
    Dframe=[]
    tau=0
    for i in inp:
        Dframe.append((np.log(i),tau,fname))
        tau+=1

    return Dframe


@jit(nopython=True)
def wolf_lyp(inp, step):



    leng=len(inp)
    leng2=int(leng/step)-1 #to compensate for forward projections at end point

    l0=np.zeros(leng2)
    li=np.zeros(leng2)



    for i in range(leng2):
        ind=int(i)
        dist, indic=nnb(inp, i*step) 
        ind1=int((i*step)+step)
        ind2=int(indic+step)
        l0[ind]=dist
        li[ind]=euc_dist1(inp[ind1],inp[ind2]) #problematic

    lyp=np.sum(np.log(li/l0))
    return lyp



#%%

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LID21.csv'




m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)

embeded=embedding.l_embed(d_body,100,1)
manif=embedding.svd_lags(embeded, 5)

lyp=wolf_lyp(manif, 5)
print(lyp)


##%% Normalization 
import matplotlib.pyplot as plt
manif2=manif.swapaxes(1,0)
norm_manf=np.zeros_like(manif2)
j=0
mn=np.mean(manif2)
st=np.std(manif2)

norm_manf=(manif2-mn)/st

#for row in manif2:
    #norm_manf[j]=(row-np.mean(row))/np.std(row)
    #j=j+1


fig=plt.figure()
plt.imshow(norm_manf, aspect= 24)
plt.xlim(0,1500)



#%%

import os
import pandas as pd 
import seaborn as sns


Dframe=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'
taus=np.arange(3,32,2)


for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname[:3] in ['SUM', 'LID','SKF']:
            print(fname)
            for dim in taus:


                case=root+'/'+fname #construct the path to file
                print(case)
                print(dim)


                m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
                m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
                tail_x=dlcp.retrivedata(case, 'tail')
                tail_y=dlcp.retrivedata(case, 'tail.1')

                #ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
                #d_body=dlcp.smooth_diff(ang_bod)
                d_ar=np.array(dlcp.ar(tail_x, tail_y, m_head_x, m_head_y))
                d_body=dlcp.smooth_diff(d_ar)                
                embeded=embedding.l_embed(d_body,100,1)
                manif=embedding.svd_lags(embeded, dim)

                if fname[:3]=='D2K':
                    lyp=wolf_lyp(manif, 15)
                else:
                    lyp=wolf_lyp(manif, 5)

                Dframe.append((lyp,dim,fname[:3]))
                #df=df+df2


#%%
df2=[]
for i in Dframe:
    df2.append((np.log(i[0]),i[1],i[2][:3]))

#%%
dframe=pd.DataFrame(df2, columns=['lyp','dim','fname']) 


#%%
import matplotlib.pyplot as plt
dframe_t=dframe[dframe['fname'].isin(['LID', 'SUM','SKF'])]


sns.lineplot(x='dim', y='lyp', hue='fname', data=dframe_t, ci=90)
plt.xlim(1.5,30)
#plt.ylim(-3, 0)

#%%
from scipy import stats
dframe_p=dframe[dframe['dim']==5]
sns.boxplot(x='fname', y='lyp',data=dframe_p)

plt.xlim(-0.5,5)





sign=stats.ttest_ind(dframe_p[dframe_p['fname']=='LID']['lyp'].values,
                        dframe_p[dframe_p['fname']=='SUM']['lyp'].values)
print(sign)



#%% lyp spectrum


case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/D2K12.csv'




m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod[::2])

embeded=embedding.l_embed(d_body,100,1)

manif=embedding.svd_lags(embeded, 25)

lyp_spec=[]
#for i in range(1, 25):
   # manif=embedding.svd_lags(embeded, i)
   # lyp_spec.append(wolf_lyp(manif, 5))


#lyp_s=np.array(lyp_spec)

manif2=manif.swapaxes(0,1)
fig=plt.figure()
plt.imshow(manif2, aspect= 18)
plt.xlim(0,900)



#%%

mn=np.mean(lyp_s)
st=np.std(lyp_s)
nrom_lyp=np.zeros_like(lyp_s)
j=0
for row in lyp_s:
    nrom_lyp[j]=(row-np.mean(row))/np.std(row)
    j=j+1

#lyp_spec=(lyp_spec-mn)/st


#%%
fig=plt.figure()
plt.imshow(lyp_s, aspect= 2)
plt.xlim(0,100)



#%%

if __name__=='__main__':
    find_ind()
    euc_dist()
    nnb()
    euc_dist1()
    wolf_lyp()
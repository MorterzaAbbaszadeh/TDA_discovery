
import numpy as np
from scipy.spatial import distance
import embedding
import dlc_tda_ppross as dlcp
from numba import jit
import os
import pandas as pd 
import seaborn as sns


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


@jit(nopython=True)
def wolf_lyp2(inp, steps, step):



    leng=len(inp)
    leng2=int(leng/step)-1 #to compensate for forward projections at end point

    l0=np.zeros(leng2)
    li=np.zeros(leng2)



    for i in range(steps):
        ind=int(i)
        dist, indic=nnb(inp, i*step) 
        ind1=int((i*step)+step)
        ind2=int(indic+step)
        l0[ind]=dist
        li[ind]=euc_dist1(inp[ind1],inp[ind2]) #problematic

    lyp=np.sum(np.log(li/l0))
    return lyp




Dframe=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'
taus=np.arange(3,32,2)


for root, dirs, files in os.walk(pth):
    for fname in files:

        
        print(fname)
        for dim in taus:


            case=root+'/'+fname #construct the path to file
            print(case)
            print(dim)


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_body=dlcp.smooth_diff(ang_bod)              
            embeded=embedding.l_embed(d_body,100,1)
            manif=embedding.svd_lags(embeded, dim)

            if manif.shape[0]>10000: #correct for high fps recording
                lyp=wolf_lyp(manif, 15)
                lyp=lyp/(manif.shape[0]*(1/140)) #normalize by total time
            else: 
                lyp=wolf_lyp(manif, 5)
                lyp=lyp/(manif.shape[0]*(1/50)) #normalize by total time

            Dframe.append((np.log(lyp),lyp,dim,fname[:3], fname[3:7]))
            print(fname[:3])


dfr=pd.DataFrame(Dframe, columns=['log_lyp', 'lyp','dim','treatment', 'animal']) 



dfr.to_csv('Lyps.csv')

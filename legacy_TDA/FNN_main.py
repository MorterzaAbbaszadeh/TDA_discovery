

import numpy as np
import pandas as pd
from numba import jit
import dlc_tda_ppross as dlcp
import matplotlib.pyplot as plt
import embedding as embd
import os
import seaborn as sns
from scipy.spatial import distance



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



def multi_d_embedding(inp,n_tau=50, tau=1):  #multi variable time_lagged coordinates (not embedding)


    chunk=len(inp[0])-(n_tau*tau)
    emb=inp[:,:chunk]
    cnt=1
    ta=int(tau)



    while cnt<n_tau:
        strt=ta*cnt
        emb=np.vstack((emb, inp[:,strt:strt+chunk]))
        cnt+=1

    return emb #dimentions is equal to 


start_ind=300
len_ind=500
dfr=[]
case_list=[]
fname_list=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data/'

for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname.endswith('.csv') and fname[:3] in ['LID', 'LES', 'SHM']:
            
            cas=root+fname #construct the path to file
            case_list.append(cas)
            fname_list.append(fname)
print(case_list)


for i in range(len(case_list)):
    case=case_list[i]
    print(fname[i])
    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)

    cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
            m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
        dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))

    embeded=embd.l_embed(d_body, 100, 1)
    u=embd.svd_lags(embeded,60)

    li_1=0
    for dim in range(1,60):
        li_2=euc_dist(u[start_ind,:dim],u[start_ind:start_ind+len_ind, :dim])
        ld=np.mean(li_2-li_1)
        li_1=li_2
        dfr.append((np.log(ld), fname_list[i][:3], fname_list[i][3:5], dim, 'body'))
        print(fname_list[i][:3])

    embeded=embd.l_embed(cos_head_ang, 100, 1)
    u=embd.svd_lags(embeded,60)

    li_1=0
    for dim in range(1,60):
        li_2=euc_dist(u[start_ind,:dim],u[start_ind:start_ind+len_ind, :dim])
        ld=np.mean(li_2-li_1)
        li_1=li_2
        dfr.append((np.log(ld), fname_list[i][:3], fname_list[i][3:5], dim, 'head'))
        print(fname_list[i][:3])



    #multiD
    r_head_x=dlcp.retrivedata(case,'headR')


    r_head_y=dlcp.retrivedata(case, 'headR.1')


    l_head_x=dlcp.retrivedata(case, 'headL')
        

    l_head_y=dlcp.retrivedata(case, 'headL.1')
        

    tail_x=dlcp.retrivedata(case, 'tail')


    tail_y=dlcp.retrivedata(case, 'tail.1')


    inp=np.array([r_head_x,r_head_y,
                    l_head_x,l_head_y,
                    tail_x,tail_y,])

    #embed and get lags by SVD
    embeded=embd.multi_d_embedding(inp,n_tau=18, tau=1)
    u=embd.svd_lags(embeded,60)

    li_1=0
    for dim in range(1,60):
        li_2=euc_dist(u[start_ind,:dim],u[start_ind:start_ind+len_ind, :dim])
        ld=np.mean(li_2-li_1)
        li_1=li_2
        dfr.append((np.log(ld), fname_list[i][:3], fname_list[i][3:5], dim, '6-Dimensional'))


df=pd.DataFrame(dfr, columns=['size','treatment', 'animal', 'dimension','Embedding'])
df.to_csv('FNN_base.csv')



import pandas as pd
import numpy as np
import scipy.signal as sgn
from scipy import stats
import embedding as emb
from numba import jit, njit
#this class has been prepared for DLC recording from above, videos taken by L.anderioli and M.abbaszadeh, Feb 2020, May2020.

#recording and pipeline characteristics
fps = 50
comp_groups =[('SKF', 'LID'), ('SKF', 'SUM'), ('SUM', 'LID')]
comp_base=[('LID', 'Base'), ('SUM', 'Base'), ('SKF', 'Base')]
times=['05', '20', '30', '40', '50', '60', '70', '80', '90']






#methods

def retrivedata(case,trg):                      #load the csv file and omit the column titles
    df=pd.read_csv(case, skiprows=[0,2])[trg].dropna()
    dta=np.array(df)
    dta_t=np.pad(dta, 50, 'edge')
    sm_dta=sgn.savgol_filter(dta_t, 45, 4) #drop na if any
    return sm_dta[50:-50]




@jit
def angs(x1,y1,x2,y2):                          #rendering vectro angles from 0 to 360
    theta=[]

    for i in range(0, len(x1)-1):
        if y2[i]-y1[i]>0:
            theta.append(np.arccos((x2[i]-x1[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32)
        elif y2[i]-y1[i]<0:                     # if the vector falls in third or forth quarters add 180 degs to result
                theta.append(180+(np.arccos((x1[i]-x2[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32))
    theta.append(theta[-1])  #compensate for shoortening of the vector
    return np.array(theta)


@njit
def cos_angs(x1,y1,x2,y2):                          #rendering vectro angles from 0 to 360
    theta=np.zeros_like(x1)
    j=0
    for i in range(0, len(x1)):
        theta[j]=(x2[i]-x1[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2))
        j=j+1
    return theta



@jit
def ar(x1,y1,x2,y2):
    ar=np.sqrt(np.square(x2-x1)+np.square(y2-y1))
    
    ar=np.pad(ar, 50, 'edge')
    sm_ar=sgn.savgol_filter(ar, 45, 4)          #savgol filter, 
    return sm_ar[50:-50]

@jit
def steps(x, y):                                #calculate the euclidian distance between positions of a point in two frames. 
    stp=[]
    for i in range(len(x)-1):
        stp.append(np.sqrt(np.square(x[i+1]-x[i])+np.square(y[i+1]-y[i])))
    stp.append(stp[-1])
    stp=np.pad(stp, 50, 'edge')                #pad the signal by its edge
    sm_stp=sgn.savgol_filter(stp, 45, 4)
    return np.array(sm_stp[50:-50])



@jit   
def thet_head(x1,y1,x2,y2,x3,y3,x4,y4):         #1:tail, 2:mid_head, 3 HeadR, 4:HeadL
    u=np.array([x2-x1, y2-y1])                  #main body vector
    v=np.array([x4-x3, y4-y3])                  #head vector
    dotp=u[0]*v[0]+u[1]*v[1]                    #dot product
    si_u=np.sqrt(u[0]**2+u[1]**2)
    si_v=np.sqrt(v[0]**2+v[1]**2)
    thet_head=np.arccos(abs(dotp/(si_u*si_v)))*57.32        #cos-1 of the absolute dot product
    return 90-thet_head



@jit   
def cos_thet_head(x1,y1,x2,y2,x3,y3,x4,y4):         #1:tail, 2:mid_head, 3 HeadR, 4:HeadL
    u=np.array([x2-x1, y2-y1])                  #main body vector
    v=np.array([x4-x3, y4-y3])                  #head vector
    dotp=u[0]*v[0]+u[1]*v[1]                    #dot product
    si_u=np.sqrt(u[0]**2+u[1]**2)
    si_v=np.sqrt(v[0]**2+v[1]**2)
    cos_thet_head=dotp/(si_u*si_v)        #cos-1 of the absolute dot product
    return cos_thet_head


@jit
def grad(theta): #gradient is the center differentiation, it is 
    grad=np.gradient(theta)
    for i in range(0,len(grad)):
        if grad[i]>30 or grad[i]<-30 :
            grad[i]=grad[i-1]
    grad=np.pad(grad, 50, 'edge')
    sm_dtheta=sgn.savgol_filter(grad, 45, 4)
    return sm_dtheta[50:-50]


@jit
def smooth_diff(theta): 
    diff=np.diff(theta)
    for i in range(0,len(diff)):
        if diff[i]>30 or diff[i]<-30 :
            diff[i]=diff[i-1]
    
    diff=np.append(diff,diff[-1])
    diff=np.pad(diff, 50, 'edge')                #pad the signal by its edge
    sm_dtheta=sgn.savgol_filter(diff, 45, 4)     #filter by a factor of 5 less than 
    return sm_dtheta[50:-50]            #report without the paddings




@jit
def get_body_manif(case):

    m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
    tail_x=retrivedata(case, 'tail')
    tail_y=retrivedata(case, 'tail.1')
    
    ang_bod=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=smooth_diff(ang_bod)


    #embed and get lags by SVD
    embeded=emb.l_embed(d_body, 100, 1)
    u=emb.svd_lags(embeded,100)

    return u



def get_head_manif(case):

    m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
    tail_x=retrivedata(case, 'tail')
    tail_y=retrivedata(case, 'tail.1')
    
    cos_head_ang=cos_thet_head(retrivedata(case,'tail'),retrivedata(case,'tail.1'),
     m_head_x,m_head_y,retrivedata(case,'headR'),retrivedata(case,'headR.1'),retrivedata(case,'headL'),
     retrivedata(case,'headL.1'))


    #embed and get lags by SVD
    embeded=emb.l_embed(cos_head_ang, 100, 1)
    u=emb.svd_lags(embeded,100)

    return u

if __name__ == '__main__':
    smooth_diff()
    grad()
    thet_head()
    steps()
    ar()
    angs()
    retrivedata()
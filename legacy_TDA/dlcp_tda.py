import pandas as pd
import numpy as np
import scipy.signal as sgn
from numba import  jit


@jit
def retrivedata(case,trg):                      #load the csv file and omit the column titles
    df=pd.read_csv(case, skiprows=[0,2])[trg].dropna()
    dta=np.array(df)
    dta_t=np.pad(dta, 150, 'edge')
    sm_dta=sgn.savgol_filter(dta_t, 149, 4) #drop na if any
    return sm_dta[150:-150]


@jit
def thet_body(x1,y1,x2,y2): 
    
                             #rendering vectro angles from 0 to 360
    theta=np.zeros_like(x1)

    for i in range(0, len(x1)-1):
        if y2[i]-y1[i]>0:
            theta[i]=np.arccos((x2[i]-x1[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32
        elif y2[i]-y1[i]<0:                     # if the vector falls in third or forth quarters add 180 degs to result
                theta[i]=180+(np.arccos((x1[i]-x2[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32)

    return theta

@jit
def smooth_diff(theta): 
    diff=np.diff(theta)
    for i in range(0,len(diff)):
        if diff[i]>50 or diff[i]<-50 :
            diff[i]=diff[i-1]
    
    pad=np.zeros(200)
    diff=np.append(diff,diff[-1])
    diff=np.append(diff,pad)
    diff=np.append(pad,diff)             #pad the signal by its edge
    sm_dtheta=sgn.savgol_filter(diff, 151, 4)     #filter by a factor of 5 less than 
    return sm_dtheta[200:-200]            #report without the paddings

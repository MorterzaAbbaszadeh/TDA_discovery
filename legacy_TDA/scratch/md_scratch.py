#%%

import numpy as np
import pandas as pd
from numba import jit
import dlc_tda_ppross as dlcp
import matplotlib.pyplot as plt
import embedding as embd
import os
import seaborn as sns
from scipy.spatial import distance



#%%

def multi_d_embedding(inp,n_tau, tau=1):  #multi variable time_lagged coordinates (not embedding)


    chunk=len(inp)-(n_tau*tau)
    emb=inp[:chunk]
    cnt=1
    ta=int(tau)
    while cnt<n_tau:
        strt=ta*cnt
        emb=np.vstack((emb, inp[strt:strt+chunk]))
        cnt+=1

    return



df=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'

for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname.endswith('.csv') and fname[:3] in (['SHM']):
            
            case=root+'/'+fname #construct the path to file


m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)

cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
        m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
    dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))


inp=np.array([d_body, cos_head_ang])

#%%

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import dlc_tda_ppross as dlcp 
from scipy import stats
import nolds as nl
import sklearn.metrics as metrics
import sklearn.feature_selection as feat
import TDA as tda
import scipy.signal as sgn


#%%Body


pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data'

for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.startswith('SHM10') :
            
            case=root+'/'+fname #construct the path to file


m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')
ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)
            



#%%

from skccm import Embed
vect=ang_bod
lag = 20
embed = [2, 20 , 30, 50, 100, 200]
for embed in embed :
    e1 = Embed(vect)
    emb_vects = e1.embed_vectors_1d(lag,embed)
    s,v,d=np.linalg.svd(emb_vects)
    print(len(v))



u=v/sum(v)
plt.plot(u[:20])


#%%

sns.lineplot(x='emb_dim', y='svd_dim', hue='lag', data=dims)


#%%

lypo=[]

stp=d_body
stp=np.pad(stp, 50, 'edge') 
sm_stp=sgn.savgol_filter(stp, 45, 4)
stp2=d_body


for i in range(30):
    lypExp=nl.lyap_r(stp2, emb_dim=3, lag=i, tau=0.02)
    lypo.append(lypExp)


plt.plot(lypo)



#%%
import scipy.signal as sgn
result = sgn.correlate(ang_bod, ang_bod, mode='full')
res=result/max(result)
plt.plot(result[7200:])
#plt.ylim(0.7,1)


# %%

#%%

'''
Controls for size and stuff
'''

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


df=[]

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'

for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname.startswith('SKF') and fname.endswith('.csv'):
            print(fname)
            case=root+'/'+fname #construct the path to file


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_body=dlcp.smooth_diff(ang_bod)



            embeded=embedding.l_embed(d_body, 100, 1)
            embeded2=np.swapaxes(embeded,0,1)

            s=svd_s(embeded2)
            #emb=np.swapaxes(u[:15], 0,1)
            df2=sns_ready(s, fname[:3])
            df=df+df2


for root, dirs, files in os.walk(pth):
    for fname in files:

        if fname.startswith('SKF') and fname.endswith('.csv'):
            print(fname)
            case=root+'/'+fname #construct the path to file


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_body=dlcp.smooth_diff(ang_bod)
            np.random.shuffle(d_body)



            embeded=embedding.l_embed(d_body, 100, 1)
            embeded2=np.swapaxes(embeded,0,1)

            s=svd_s(embeded2)
            #emb=np.swapaxes(u[:15], 0,1)
            df2=sns_ready(s, 'shuff_'+fname[:3])
            df=df+df2







dframe=pd.DataFrame(df, columns=['delta','tau','fname']) 
dframe.to_csv('control_dimension_s_groups.csv')

#%%

sns.lineplot(x='tau', y='delta', hue='fname', data=dframe, ci=90)
sns.despine()



# %% Correct embedding



m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)




embeded=embedding.l_embed(d_body, 4, 20)


#%% Shuffled embedding


m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')

ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)
np.random.shuffle(d_body)



sh_embeded=embedding.l_embed(d_body, 4, 20)




#%% Visualization
 
plt.plot(embeded[0, :300], embeded[1, :300])
plt.plot(sh_embeded[0, :200], sh_embeded[1, :200], alpha=0.5)



# %%
plt.plot(embeded[3, :1500])
plt.plot(sh_embeded[3, :1500], alpha=0.5)
# %%

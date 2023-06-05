
#%%

# get significance of the dimentiosn



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
        #Dframe.append((np.log(i),tau,fname)) #logarithmic
        Dframe.append((i,tau,fname))
        tau+=1

    return Dframe


@jit
def sns_ready_c(inp, fname):
    Dframe=[]
    i_dim=len(inp)
    for i in range(i_dim):
        #Dframe.append((np.log(i),tau,fname)) #logarithmic
        Dframe.append((np.sum(inp[:i]),i,fname))


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

        if fname.endswith('.csv'):
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
            #df2=sns_ready_c(s, fname[:3])
            df2=sns_ready_c(s, 'Shuffle')
            df=df+df2


dframe=pd.DataFrame(df, columns=['delta','tau','fname']) 
dframe.to_csv('clum_dimension_s_shuff2.csv')


#%%

import seaborn as sns
dims_df=dframe[dframe['tau']<=7]


sns.boxplot(x='tau', y='delta', hue='fname', data=dims_df.loc[dims_df['fname'].isin(['LES', 'LID', 'SHM'])])


# %%

dims_df[dims_df['fname']=='SHM'].sum()['delta']/8
# %%


dims_df=dframe[dframe['tau']<=7]


sns.boxplot(x='tau', y='delta', hue='fname', data=dims_df)
# %%

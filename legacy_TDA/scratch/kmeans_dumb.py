#%% k-means


import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dlc_tda_ppross as dlcp
from scipy import stats
import sklearn.metrics as metrics
import embedding

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_manif(case, dim):
    



    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)


    embeded=embedding.l_embed(d_body,100,1)
    manif=embedding.svd_lags(embeded, dim)

    return manif


#%%

dim=7

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data/LES3.csv'
cause=get_manif(case, dim)





#%%

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'
dim=5


score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LES'):
            case=root+'/'+fname
            print(fname)

            cause=get_manif(case, dim)
           
            for i in range(2,50):
                kmeans = KMeans(n_clusters=i).fit(cause)
                sc_r=metrics.silhouette_score(cause, kmeans.labels_)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

#%%
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))


ax=sns.lineplot(x='n_clusters', y='score',data=scores, ci=95)
#plt.ylim(0,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()




# %%

pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/tda_data'
dim=15

score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LES'):
            case=root+'/'+fname
            print(fname)

            cause=get_manif(case, dim)
           
            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(cause)
                label1=model1.predict(cause)
                sc_r=metrics.silhouette_score(cause, label1)
                score.append((fname[3:5], sc_r, i, 'LID'))



scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])


#%%
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))


ax=sns.lineplot(x='n_clusters', y='score',data=scores, ci=95)
#plt.ylim(0,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()

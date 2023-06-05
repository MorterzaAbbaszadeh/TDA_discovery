

#%%

import numpy as np
import pandas as pd
import dlc_tda_ppross as dlcp 
from numba import jit
import scipy.spatial as space 
import matplotlib.pyplot as plt
import os
import seaborn as sns


from scipy.optimize import differential_evolution, minimize


@jit
def opt_nonu_emb(inp, tau_l, taus, delta,eps):


   
    dim=len(tau_l)
    tau_m=np.amax(tau_l)
    leng=len(inp)
    chunk=int((dim+10)*tau_m)

    max_cap=[]


    tau_count=0
    for tau in tau_l:
        cap=[]
        strt=int(np.sum(tau_l[:tau_count]))

        if tau_count==0:
            embeded=inp[:(leng-chunk)]
        else:
            embeded=np.vstack((embeded,inp[strt:(strt+(leng-chunk))])) #stack the next dimension
        tau_count+=1

        for t in range(1,taus):
            embeded_a=np.vstack((embeded,inp[(strt+t):(strt+t+(leng-chunk))])) 


            embeded_l=embeded_a[-1].reshape(-1,1)
            araydelta=space.distance_matrix(embeded_l[5:6], embeded_l)
            #num_delta=len(araydelta[araydelta<delta])
            embeded_a=np.transpose(embeded_a)
            embeded_a=embeded_a[np.where(araydelta<delta)[1]]
            araydelta=space.distance_matrix(embeded_a[5:6], embeded_a)
            num_eps=len(araydelta[araydelta<=(delta*eps)])


            
            cap.append(num_eps/(leng-chunk))
        max_cap.append(np.amax(cap))
        



    return 1/sum(max_cap)


@jit
def eval_nonu_emb(fname, inp, tau_l, taus, delta,eps):


    tau_l=tau_l.astype(int)
    dim=len(tau_l)
    tau_m=np.amax(tau_l)
    leng=len(inp)
    chunk=int((dim+10)*tau_m)

    df=[]


    tau_count=0
    for tau in tau_l:
        cap=[]
        strt=int(np.sum(tau_l[:tau_count]))

        if tau_count==0:
            embeded=inp[:(leng-chunk)]
        else:
            embeded=np.vstack((embeded,inp[strt:(strt+(leng-chunk))])) #stack the next dimension
        tau_count+=1
        embeded_a=embeded
        embeded_l=embeded_a[-1].reshape(-1,1)


        araydelta=space.distance_matrix(embeded_l[5:6], embeded_l)
        #num_delta=len(araydelta[araydelta<delta])
        embeded_a=np.transpose(embeded_a)
        embeded_a=embeded_a[np.where(araydelta<delta)[1]]
        araydelta=space.distance_matrix(embeded_a[5:6], embeded_a)
        num_eps=len(araydelta[araydelta<=(delta*eps)])
        df.append((fname[:7],tau_count, tau, num_eps/(leng-chunk)))
    
    return df





#%%


pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data'

for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.startswith('SUM12') :
            
            case=root+'/'+fname #construct the path to file


m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
tail_x=dlcp.retrivedata(case, 'tail')
tail_y=dlcp.retrivedata(case, 'tail.1')
ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
d_body=dlcp.smooth_diff(ang_bod)




#%% optimization

bodydiff=d_body.reshape(-1,1)
delta=np.amax(space.distance_matrix(bodydiff[5:6], bodydiff))
eps=1


bound=[(1,50)]*20

guess=np.random.rand(10)*20+1

tres=differential_evolution(opt_nonu_emb, bound, args=(d_body, 80, delta, 1 ),maxiter=20,popsize=5)
print(tres)
#%%



tau_l=[25]*10
scor=eval_nonu_emb(fname, d_body, tres.x.astype(int), 80, 2, 1)    #eval_nonu_emb(fname, inp, tau_l, taus, delta,eps):

corrs=pd.DataFrame(scor, columns=['treatment','dimention', 'tau','capture']) #(fname[:7],tau_count, tau, num_eps/(leng-chunk))
sns.lineplot(x='dimension',y='capture', hue='tau', data=corrs ) #, hue='dimention'
#plt.ylim(0,1)


# %%


array([31.59326614, 17.26989906, 24.72489512, 20.55591468, 37.42200683,
       37.34784309, 31.75821142, 30.08418835, 23.43650076, 14.82729793])
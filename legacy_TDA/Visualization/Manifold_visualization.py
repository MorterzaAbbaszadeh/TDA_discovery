
#%%


import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.spatial import distance
import dlc_tda_ppross as dlcp
import embedding
import matplotlib.pyplot as plt
import seaborn as sns

def get_body_manif(case, dim):
    



    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)



    #dim=25
    #Shuffling
    #d_body=np.random.shuffle(d_body)
    #np.random.shuffle(d_body)



    #SVD Embedding
    embeded=embedding.l_embed(d_body,100,1)
    manif=embedding.svd_lags(embeded, dim)

    return manif



def get_head_manif(case, dim):


    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2

    cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
     m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
     dlcp.retrivedata(case,'headL.1'))


    embeded=embedding.l_embed(cos_head_ang,100,1)
    manif=embedding.svd_lags(embeded, dim)
    return manif



def get_shuff_manif(case, dim):
    



    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)



    #dim=25
    #Shuffling
    #d_body=np.random.shuffle(d_body)
    np.random.shuffle(d_body)



    #SVD Embedding
    embeded=embedding.l_embed(d_body,100,1)
    manif=embedding.svd_lags(embeded, dim)

    return manif

def get_tot_manif(case, dim):


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
    embeded=embedding.multi_d_embedding(inp,n_tau=18, tau=1)
    u=embedding.svd_lags(embeded,dim)

    return u


def combined_manif(case, dim):

    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)

    cos_head_ang=dlcp.cos_thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
        m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),dlcp.retrivedata(case,'headL'),
        dlcp.retrivedata(case,'headL.1'))


    inp=np.array([d_body,cos_head_ang])
    #embed and get lags by SVD
    embeded=embedding.multi_d_embedding(inp,n_tau=100, tau=1)
    u=embedding.svd_lags(embeded,dim)

    return u




#%% The main body manifold visualization 


case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SHM21.csv'

manif=get_body_manif(case, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(manif[:3000,0],manif[:3000,1],linewidth=0.7, color='black')

ax.set_xlim(-20,20)
ax.set_ylim(-22,20)

ax.set_xlabel('Dimension 1', fontsize='large')
ax.set_ylabel('Dimension 2', fontsize='large')
sns.despine()


#%% The main body manifold visualization 


case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SHM21.csv'

h_manif=get_head_manif(case, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(h_manif[:3000,0],h_manif[:3000,1],linewidth=0.7, color='black')

ax.set_xlim(-20,20)
ax.set_ylim(-22,20)

ax.set_xlabel('Dimension 1', fontsize='large')
ax.set_ylabel('Dimension 2', fontsize='large')
sns.despine()

#%% The main body manifold visualization 


case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SHM21.csv'

t_manif=combined_manif(case, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t_manif[:3000,0],t_manif[:3000,1],linewidth=0.7, color='black')

ax.set_xlim(-20,20)
ax.set_ylim(-22,20)

plt.title('Original Data', fontsize='large')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
sns.despine()



'''
heat maps
'''
#%%
cmap = plt.cm.magma


fig = plt.figure()
manif=np.flip(manif.T, axis=0)
manif=manif/np.sum(manif)


plt.imshow(manif[-12:,:500],vmin=-0.0005, vmax=0.0005, cmap=cmap, aspect=24)
sns.despine(top=True, bottom=True, left=True, right=True)
plt.xticks([])
plt.yticks([])
plt.title('Body', fontsize='large')

#%%



#%%


fig = plt.figure()
h_manif=np.flip(h_manif.T, axis=0)
h_manif=h_manif/np.sum(h_manif)

plt.imshow(h_manif[-12:,:500],vmin=-0.0005, vmax=0.0005, cmap=cmap, aspect=24)

sns.despine(top=True, bottom=True, left=True, right=True)
plt.xticks([])
plt.yticks([])
plt.title('Head', fontsize='large')

#%%


fig = plt.figure()
t_manif=np.flip(t_manif.T, axis=0)
t_manif=t_manif/np.sum(t_manif)

plt.imshow(t_manif[-12:,:500],vmin=-0.0005, vmax=0.0005,cmap=cmap, aspect=24)

sns.despine(top=True, bottom=True, left=True, right=True)
plt.xticks([])
plt.yticks([])

plt.plot([0, 50], [12,12], color='black')
plt.text(0,13, '1 Sec.')

plt.plot([-7,-7], [9,11], color='black')
plt.text(-27,11, '2 Dimensions', rotation='vertical')


plt.title('Combined', fontsize='large')



#%% Shuffled input visualization


sh_manif=get_shuff_manif(case, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sh_manif[:500,0],sh_manif[:500,1],linewidth=0.7, color='black')

ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
plt.title('Scrambled Data', fontsize='large')

ax.set_xlabel('Dimension 1', fontsize='large')
ax.set_ylabel('Dimension 2', fontsize='large')
sns.despine()


#%%

fig, ax=plt.subplots(2,1)


ax[0].plot(manif[:500,1],linewidth=.7,color='black')
ax[0].set_ylim(-10,10)

ax[1].plot(sh_manif[:500,1],linewidth=0.7, color='black')
ax[1].set_ylim(-10,10)
sns.despine()


#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=manif[:2000,0],ys=manif[:2000,1],zs=manif[:2000,2],alpha=1)


#%%

refr= []
for i in range(50):
    refr.append([(50-i)*0.02]*10)
refr=np.array(refr)


plt.imshow(refr,cmap=cmap, aspect=2)
sns.despine(top=True, bottom=True, left=True, right=True)
ax=plt.gca()
ax.set_yticks([1, 49])
ax.set_yticklabels(['0.005', '-0.005'], fontsize='large')
plt.xticks([])

plt.text(-15, 30, 'Z-Score', fontsize=18, rotation=90)



# %% LID

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/LID21.csv'

manif=get_body_manif(case, 50)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(manif[:1500,0],manif[:1500,1],linewidth=0.7, color='violet')

#ax.set_xlim(-50,50)
#ax.set_ylim(-50,50)
ax.set_xticks([])
ax.set_yticks([])

plt.title('LID', fontsize='large', color='violet')

ax.set_xlabel('')
ax.set_ylabel('')

plt.plot([-5,5],[-30,-30], color='black')
plt.text(-5,-35, '10 units')
sns.despine(top=True, bottom=True, left=True, right=True)


# %% SKF

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SKF17.csv'

manif=get_body_manif(case, 50)



#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(manif[:1500,0],manif[:1500,1],linewidth=0.7, color='blue')

#ax.set_xlim(-50,50)
#ax.set_ylim(-50,50)
ax.set_xticks([])
ax.set_yticks([])

plt.title('SKF', fontsize='large', color='blue')

ax.set_xlabel('')
ax.set_ylabel('')
sns.despine(top=True, bottom=True, left=True, right=True)

plt.plot([-35,-25],[-30,-30], color='black')
plt.text(-34,-34, '10 units')

# %% SUM

case='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/Visualization/pilot_data/SUM12.csv'

manif=get_body_manif(case, 50)



#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(manif[:1500,0],manif[:1500,1],linewidth=0.7, color='darkgreen')

#ax.set_xlim(-50,50)
#ax.set_ylim(-50,50)
ax.set_xticks([])
ax.set_yticks([])

plt.title('Sumanirole', fontsize='large', color='darkgreen')

ax.set_xlabel('')
ax.set_ylabel('')
sns.despine(top=True, bottom=True, left=True, right=True)

plt.plot([-27,-17],[-16,-16], color='black')
plt.text(-24,-18, '10 units')
# %%

import numpy as np 
from multiprocessing import Pool
import embedding as emb 
import os
import pickle
import dlc_tda_ppross as dlcp

def func(case):
    #get the coordinate time series
    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2        #get bodypart signals
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1')

    #calculate the 
    ang_bod=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
    d_body=dlcp.smooth_diff(ang_bod)

    #embed and get lags by SVD
    embeded=emb.l_embed(d_body, 100, 1)
    u=emb.svd_lags(embeded,100)


    #savedestination

    l=case.rfind('/') #find the last /
    with open(case[:l+6]+'.pickle', 'wb') as f:
        pickle.dump(u, f)






#first get a list of csv files
pth='/home/morteza/dlc_projects/Analysis/Currencodes/TDA/pilot_data'
csv_list=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('csv') :
            case=root+'/'+fname #construct the path to file
            csv_list.append(case)






if __name__=='__main__':
    with Pool() as pool:
        pool.map(func, csv_list)



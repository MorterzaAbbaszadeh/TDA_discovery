
import numpy as np



def nl_embed(inp, v, tau, tau_max, n_final):  #length of v should be the same as the length of tau
    
    
    chunk=len(inp)-(tau_max*n_final)


    emb=inp[:chunk]
    strt=0
    for t in tau:
        strt+=t
        emb=np.vstack((emb, inp[strt:strt+chunk]))

    
    return np.array(emb)
        



def l_embed(inp,n_tau, tau): #time_lagged coordinates (not embedding)
    
    
    chunk=len(inp)-(n_tau*tau)
    emb=inp[:chunk]
    cnt=1
    ta=int(tau)
    while cnt<n_tau:
        strt=ta*cnt
        emb=np.vstack((emb, inp[strt:strt+chunk]))
        cnt+=1

    
    return np.array(emb)



def multi_d_embedding(inp,n_tau=50, tau=1):  #multi variable time_lagged coordinates (not embedding)


    chunk=len(inp[0])-(n_tau*tau)
    emb=inp[:,:chunk]
    cnt=1
    ta=int(tau)



    while cnt<n_tau:
        strt=ta*cnt
        emb=np.vstack((emb, inp[:,strt:strt+chunk]))
        cnt+=1

    return emb #dimentions is equal to 




def svd_lags(embeded,dims):
    
    def swap(inp):
        i=inp.shape[0]
        j=inp.shape[1]
        v=np.zeros((j,i))

        for i in range(len(inp)):
            for j in range(len(inp[1])):
                v[j,i]=inp[i,j]
        return v


    embeded2=swap(embeded)


    _,_, v=np.linalg.svd(embeded2)

    u2=np.dot(embeded2, v.T)
    u2=swap(u2)
    u3=swap(u2[:dims])


    return u3


if __name__ == "__main__":
    svd_lags()
    multi_d_embedding()
    l_embed()
'''

General utility functions 

'''
import numpy as np 

def ijl_order(i_k, j_k, l_k, typ='GM'): 
    ''' triangle configuration ordering, returns indices
    '''
    i_bq = np.arange(len(i_k))
    if typ == 'GM': # same order as Hector's
        i_bq_new = [] 

        l_usort = np.sort(np.unique(l_k))
        for l in l_usort: 
            j_usort = np.sort(np.unique(j_k[l_k == l]))
            for j in j_usort: 
                i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                for i in i_usort: 
                    i_bq_new.append(i_bq[(i_k == i) & (j_k == j) & (l_k == l)])
    else: 
        raise NotImplementedError
    return np.array(i_bq_new)


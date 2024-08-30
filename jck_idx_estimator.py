import numpy as np

def jck_idx_estimator(R_d):
    n, m = len(R_d), len(R_d) - 1
    def jk(A):
        c = A[:-1,-1]
        Rxx = A[:-1,:-1]
        jk_squared = c@(np.linalg.pinv(Rxx))@c
        return np.sqrt(jk_squared)
    jk_idx = []

    jk_temp = np.zeros((m,))
    for i in range(m):
        idx_jk = np.array([i,n-1])
        Rtemp = R_d[idx_jk][:,idx_jk]
        jk_temp[i] = jk(Rtemp)
    jk_sort = np.argsort(jk_temp)[::-1]
    jk_idx.append(jk_sort[0])

    for j in range(m-1):
        jk_temp = np.zeros((m,))
        for i in range(m):
            if i in jk_idx:
                pass
            else:
                idx_jk = np.zeros((j+3,), dtype = 'int')
                idx_jk[:j+1] = jk_idx[:j+1]
                idx_jk[j+1] = i
                idx_jk[-1] = n-1
                Rtemp = R_d[idx_jk][:,idx_jk]
                jk_temp[i] = jk(Rtemp)
        jk_sort = np.argsort(jk_temp)[::-1]
        jk_idx.append(jk_sort[0])
    return jk_idx
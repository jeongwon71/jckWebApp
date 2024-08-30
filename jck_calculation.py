import numpy as np
from jck_idx_estimator import jck_idx_estimator

def jck_calc(calc_mean, calc_cov, meas_mean, meas_cov):
    pred_k, pred_v = list([calc_mean[0]]), list([calc_cov[0,0]])
    for i in range(len(meas_mean)):
        pred_k.append(calc_mean[0] - calc_cov[0,1:i+2]@np.linalg.pinv(calc_cov[1:i+2,1:i+2]+meas_cov[:i+1,:i+1])@(calc_mean[1:i+2]-meas_mean[:i+1]))
        pred_v.append(calc_cov[0,0] - calc_cov[0,1:i+2]@np.linalg.pinv(calc_cov[1:i+2,1:i+2]+meas_cov[:i+1,:i+1])@calc_cov[0,1:i+2])
    return {"pred_k": pred_k, "pred_u": np.sqrt(pred_v)}

def jck_versus_num_exp(mu_c, mu_e, Sig_c, Sig_e):
    methods = ['ck-ascending','ck-descending','optimal(jk)']
    results = {}
    for method in methods:
        if method == 'ck-ascending':
            temp = np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_c))))@Sig_c@np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_c))))
            idx = np.argsort(temp[-1,:-1])
            idx_ = np.concatenate((np.array([-1]),idx))
            results[method] = jck_calc(mu_c[idx_], Sig_c[idx_][:,idx_], mu_e[idx], Sig_e[idx][:,idx])
        elif method == 'ck-descending':
            temp = np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_c))))@Sig_c@np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_c))))
            idx = np.argsort(temp[-1,:-1])
            idx = idx[::-1]
            idx_ = np.concatenate((np.array([-1]),idx))
            print(idx_)
            results[method] = jck_calc(mu_c[idx_], Sig_c[idx_][:,idx_], mu_e[idx], Sig_e[idx][:,idx])
        else:
            Sig_d = np.copy(Sig_c)
            Sig_d[:-1,:-1] += Sig_e
            temp = np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_d))))@Sig_d@np.linalg.inv(np.diag(np.sqrt(np.diag(Sig_d))))
            idx = jck_idx_estimator(temp)
            idx_ = np.concatenate((np.array([-1]),idx))
            print(idx_)
            results[method] = jck_calc(mu_c[idx_], Sig_c[idx_][:,idx_], mu_e[idx], Sig_e[idx][:,idx])
    return results


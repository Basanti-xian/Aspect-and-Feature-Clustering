from requests import get
from math import isinf
from joblib import Parallel,delayed
import numpy as np
from pprint import pprint

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"
def sss(i,j, s1, s2, type='relation', corpus='webbase'):
    return 0
    if i > j:
        return 0
    if i == j:
        return 1
    try:
        response = get(sss_url,
                       params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
        print s1, s2, float(response.text.strip())
        if isinf(float(response.text.strip())):
            return 0.0
        else:
            return float(response.text.strip())
    except:
        print 'Error in getting similarity for %s: %s' % ((s1, s2), response)
        return 0.0


def get_G_semantic_matrix(data):
    G = []
    for i,wi in enumerate(data):
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        # row = Parallel(n_jobs=20)(delayed(sss)(i,j,wi,wj)
        #
        row = []
        for j, wj in enumerate(data):
            row.append(sss(i,j,wi,wj))
        G.append(row)
    G = np.array(G)
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            if i > j:
                G[i,j] = G[j,i]
    # pprint (G)
    # raw_input()
    return G
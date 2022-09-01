import ot
import pickle
import time
from treeOT import *

dataname = 'twitter'

#Load data
data = np.load('dataset/%s.npz' % dataname,allow_pickle=True)
X = data['X']
X_all = data['X_all']
A = data['A']

with open('dataset/%s_indices.pkl' % dataname, 'rb') as f:
    indices = pickle.load(f)

#n_leaf: number of word vectors
#d: the dimensionality of the word vector
n_leaf,d = X_all.shape

#Number of documents
n = len(X[0])

#Cost matrix (this is mainly for EMD computation)
M_all = ot.dist(X_all,X_all,metric='euclidean')

#Construct tree
# 'cluster': cluster tree
# 'quad': Quadtree
# n_slice: Number of tree slice
# lam: The regularization parameter
n_slice = 1
lam = 0.001
ctree = treeOT(X_all / M_all.max(), method='cluster', lam=lam, n_slice=n_slice)

#Test
n_test_sample = 1000
np.random.seed(0)
ind1 = np.random.randint(0,n,n_test_sample)
ind2 = np.random.randint(0,n,n_test_sample)

score_WD      = np.zeros(n_test_sample)
A_full = np.zeros((n_leaf,n_test_sample))
B_full = np.zeros((n_leaf,n_test_sample))

elapsed_time_emd = 0
for ii in range(n_test_sample):

    #Calculating mass
    a = A[0][ind1[ii]][0]
    b = A[0][ind2[ii]][0]
    a = a/a.sum()
    b = b/b.sum()

    # Compute the original Wasserstein distance
    M = ot.dist(X[0][ind1[ii]].transpose(), X[0][ind2[ii]].transpose(), metric='euclidean')

    start = time.time()
    score_WD[ii] = ot.emd2(a, b, M / M_all.max())
    elapsed_time_emd += time.time() - start

    #Preparing the probability mass for TWD
    ind1_ = indices[0][ind1[ii]]
    ind2_ = indices[0][ind2[ii]]
    A_full[ind1_,ii] = a
    B_full[ind2_,ii] = b

A_full = csr_matrix(A_full.astype(np.float32))
B_full = csr_matrix(B_full.astype(np.float32))

start = time.time()
result_TWD = ctree.pairwiseTWD(A_full,B_full)
elapsed_time_twd = time.time() - start

print('Error: %f' % (np.abs(score_WD-result_TWD).mean()))
print('R2: %f' % (np.corrcoef(score_WD,result_TWD)[0][1]))
print('Time: EMD (%f), TWD (%f)' % (elapsed_time_emd,elapsed_time_twd))

import numpy as np
from treelib import Tree
import copy
from tqdm import tqdm
import random
import spams
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
import networkx as nx
import joblib
import faiss

class treeOT():
    def __init__(self, X, method='cluster', cluster_type='kmeans', lam=0.0001,nmax=100000, k=5, d=6, n_slice=1, debug_mode=False,is_sparse=True,is_leaf=True):
        """
         Parameter
         ----------
         X :
             a set of supports
         method :
             'cluster' (clustering tree) or 'quad' (quadtree)
         k : int
             a number of child nodes
         d : int
             depth of a tree
         n_slice : int
             the number of sampled trees
         lam: float
             the regularization parameter of Lasso
         nmax: int
             the number of training samples for Lasso
         """

        self.n_slice = n_slice

        for i in tqdm(range(n_slice)):

            if method=='quad': #Quadtree
                np.random.seed(i)
                tree = self.build_quadtree(X, random_shift=True, width=None, origin=None)
                print("build done")
            else: #Clustering tree
                random.seed(i)
                tree = self.build_clustertree(X, k, d, debug_mode=debug_mode,cluster_type=cluster_type)
                print("build done")

            Bsp = self.get_B_matrix(tree,X,is_leaf=is_leaf)


            if is_sparse:
                wv = self.calc_weight_sparse(X, Bsp, lam=lam, nmax=nmax)
            else:
                wv = self.calc_weight(X,Bsp.toarray(),lam=lam,nmax=nmax)

            if i == 0:
                wB = Bsp.multiply(wv)
            else:
                wB = sparse.vstack([wB,Bsp.multiply(wv)])


        self.wB = wB


    def incremental_farthest_search(self, points, remaining_set, k, debug_mode=False):
        n_points = len(remaining_set)
        remaining_set = copy.deepcopy(remaining_set)

        if not debug_mode:
            # random.seed(0)
            solution_set = [remaining_set[random.randint(0, n_points - 1)]]
        else:
            solution_set = [remaining_set[0]]
        remaining_set.remove(solution_set[0])

        for i in range(k - 1):

            distance_list = []

            for idx in remaining_set:
                in_distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
                distance_list.append(min(in_distance_list))

            sol_idx = remaining_set[np.argmax(distance_list)]
            remaining_set.remove(sol_idx)
            solution_set.append(sol_idx)

        return solution_set


    def distance(self, A, B):
        return np.linalg.norm(A - B)


    def grouping(self, points, remaining_set, solution_set):
        np.random.seed(0)
        n_points = len(points)
        remaining_set = copy.deepcopy(remaining_set)

        group = []
        for _ in range(len(solution_set)):
            group.append([])

        for idx in remaining_set:
            distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
            group_idx = np.argmin(distance_list)
            group[group_idx].append(idx)

        return group



    def clustering(self, points, remaining_set, k, debug_mode=False,cluster_type='fs'):

        if cluster_type=='fs':
            solution_set = self.incremental_farthest_search(points, remaining_set, k, debug_mode=debug_mode)
            return self.grouping(points, remaining_set, solution_set)
        else:
            kmeans = faiss.Kmeans(points.shape[1], k)
            kmeans.train(points[remaining_set])
            D,I=kmeans.index.search(points[remaining_set],1)
            solution_set = []
            for kk in range(k):
                solution_set.append(np.where(I.flatten()==kk)[0].tolist())
            
            return solution_set
       


    def _build_clustertree(self, X, remaining_set, k, d, debug_mode=False,cluster_type='fs'):
        tree = Tree()
        tree.create_node(data=None)

        if len(remaining_set) <= k or d == 1:
            for idx in remaining_set:
                tree.create_node(parent=tree.root, data=idx)
            return tree

        groups = self.clustering(X, remaining_set, k, debug_mode=debug_mode,cluster_type=cluster_type)
        # print(groups)
        for group in groups:
            if len(group) == 1:
                tree.create_node(parent=tree.root, data=group[0])
            else:
                subtree = self._build_clustertree(X, group, k, d - 1, debug_mode=debug_mode)
                tree.paste(tree.root, subtree)
        return tree


    def build_clustertree(self, X, k, d, debug_mode=False,cluster_type='kmeans'):
        """
        k : the number of child nodes
        d : the depth of the tree
        """
        remaining_set = [i for i in range(len(X))]
        return self._build_clustertree(X, remaining_set, k, d, debug_mode=debug_mode,cluster_type=cluster_type)

    def _build_quadtree(self, X, origin, remaining_idx, width):
        d = X.shape[1]  # dimension
        m = len(remaining_idx)  # number of samples (i.e., support size)

        tree = Tree()
        tree.create_node(data=None)

        loc = np.zeros(m).tolist()

        # divide the hypercube, and obtain which hypercube a point belong to.
        for i in range(len(remaining_idx)):
            for j in range(d):
                if X[remaining_idx[i]][j] > origin[j]:
                    loc[i] += 2 ** j

        child = list(set(loc))
        child_set = [[] for _ in range(len(child))]
        origin_set = []

        for i in range(len(child)):
            new_origin = np.zeros_like(origin)
            for j in range(d):
                if int(child[i]) & (2 ** j) != 0:
                    new_origin[j] = copy.deepcopy(origin[j]) + width / 2.0
                else:
                    new_origin[j] = copy.deepcopy(origin[j]) - width / 2.0
            origin_set.append(new_origin)

        for i in range(m):
            child_set[child.index(loc[i])].append(remaining_idx[i])

        for i in range(len(child)):
            if len(child_set[i]) == 1:
                tree.create_node(parent=tree.root, data=child_set[i][0])
            else:
                subtree = self._build_quadtree(X, origin_set[i], child_set[i], width / 2.0)
                tree.paste(tree.root, subtree)

        return tree

    def build_quadtree(self, X, random_shift=True, width=None, origin=None):
        """
        Assume that X[i] in [0, width]^d.
        """
        #np.random.seed(0)
        if random_shift:
            # check the assumption.
            if np.min(X) < 0:
                print("Warn : Assumption")
                X = X - np.min(X)
            elif np.min(X) != 0:
                print("Warn : Assumption")

            width = np.max(X)
            origin = np.random.uniform(low=0.0, high=width, size=X.shape[1])

        remaining_idx = [i for i in range(X.shape[0])]

        return self._build_quadtree(X, origin, remaining_idx, width)


    def gen_matrix(self, tree, X):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in = n_node - n_leaf
        D1 = np.zeros((n_in, n_in))
        D2 = np.zeros((n_in, n_leaf))

        in_node = [node.identifier for node in tree.all_nodes() if node.data == None]

        for node in tree.all_nodes():
            # check node is leaf or not
            if node.data is not None:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                D2[parent_idx, node.data] = 1.0
            elif node.identifier == tree.root:
                continue
            else:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                node_idx = in_node.index(node.identifier)
                D1[parent_idx, node_idx] = 1.0
        return D1, D2

    def get_B_matrix(self, tree, X,is_leaf=True):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in   = n_node - n_leaf

        #B = np.zeros((n_node,n_leaf))

        in_node   = [node.identifier for node in tree.all_nodes() if node.data == None]
        in_node_index = [ii for ii in range(n_in)]
        leaf_node = [node.identifier for node in tree.all_nodes() if node.data != None]
        leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]
        #leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]
        path_leaves = tree.paths_to_leaves()

        n_edge = 0
        for path in path_leaves:
            n_edge += len(path)
        col_ind = np.zeros(n_edge)
        row_ind = np.zeros(n_edge)
        cnt = 0
        for path in path_leaves:
            # check node is leaf or not
            leaf_index = leaf_node_index[leaf_node.index(path[-1])]
            #B[leaf_index,leaf_index] = 1.0
            col_ind[cnt] = leaf_index
            row_ind[cnt] = leaf_index
            cnt += 1
            for node in path[:-1]:
                in_index = in_node_index[in_node.index(node)] + n_leaf
                #B[in_index,leaf_index] = 1.0
                col_ind[cnt] = leaf_index
                row_ind[cnt] = in_index
                cnt+=1

        B = sparse.csc_matrix((np.ones(n_edge), (row_ind, col_ind)), shape=(n_node, n_leaf), dtype='float32')
        
        #Remove leaf node to reduce number of parameters
        if is_leaf == False:
            B = B[n_leaf:,:]
        return B

    def get_B_matrix_networkx(T, root_node, nodes_tree=[]):
        """
        Usage:
        #G is a Graph (networkx format)

        T = nx.dfs_tree(G, root_node)
        B,nodes_tree = get_matrix_networkx(G,root_node,nodes_tree=labels)
        Bsp = sparse.csc_matrix(B)

        :param root_node:
        :return: B, nodes_tree
        """
        if len(nodes_tree) == 0:
            nodes_tree = list(T.nodes())

        dict_nodes = {}
        ii = 0
        for node in nodes_tree:
            dict_nodes[node] = ii
            ii += 1

        B = np.zeros((len(nodes_tree), len(nodes_tree)))
        ii = 0
        for node in nodes_tree:
            node_current = node
            B[dict_nodes[node_current], ii] = 1
            B[dict_nodes[root_node], ii] = 1
            while node_current is not root_node:
                try:
                    node_current = list(T.predecessors(node_current))[0]
                    B[dict_nodes[node_current], ii] = 1
                except:
                    node_current = root_node
            ii += 1

        return B, nodes_tree

    def calc_weight(self, X, B, lam=0.001, seed=0, nmax=100000):

        n_leaf, d = X.shape
        random.seed(seed)

        dz = B.shape[0]

        np.random.seed(seed)
        ind1 = np.random.randint(0, n_leaf, nmax)
        ind2 = np.random.randint(0, n_leaf, nmax)

        c_all = np.zeros((nmax, 1))
        Z_all = np.zeros((dz, nmax))

        for ii in range(nmax):
            c_all[ii] = np.linalg.norm(X[ind1[ii], :] - X[ind2[ii], :], ord=2)
            Z_all[:, ii] = B[:, ind1[ii]] + B[:, ind2[ii]] - 2 * (B[:, ind1[ii]] * B[:, ind2[ii]])

        n_sample = nmax
        c = np.asfortranarray(c_all[:n_sample, 0].reshape((n_sample, 1)), dtype='float32')
        Z = np.asfortranarray(Z_all[:, :n_sample].transpose(), dtype='float32')
        Zsp = sparse.csc_matrix(Z)

        # Solving nonnegative Lasso
        param = {'numThreads': -1, 'verbose': True,
                 'lambda1': lam, 'it0': 10, 'max_it': 2000, 'tol': 1e-3, 'intercept': False,
                 'pos': True}

        param['loss'] = 'square'
        param['regul'] = 'l1'

        W0 = np.ones((Z.shape[1], c.shape[1]), dtype='float32', order="F")

        (W, optim_info) = spams.fistaFlat(c, Zsp, W0, True, **param)

        return W

    def calc_weight_in(self,X, Bsp, ind1, ind2):
        n = len(ind1)
        c_tmp = np.zeros(n)
        for ii in range(n):
            c_tmp[ii] = np.linalg.norm(X[ind1[ii], :] - X[ind2[ii], :], ord=2)
            tmp = Bsp[:, ind1[ii]] + Bsp[:, ind2[ii]] - 2 * (Bsp[:, ind1[ii]].multiply(Bsp[:, ind2[ii]]))

            if ii == 0:
                row_ind = tmp.indices
                col_ind = np.ones(len(row_ind)) * ii
                data = tmp[row_ind].toarray().flatten()
            else:
                row_ind_tmp = tmp.indices
                col_ind_tmp = np.ones(len(row_ind_tmp)) * ii
                row_ind = np.concatenate((row_ind, row_ind_tmp))
                col_ind = np.concatenate((col_ind, col_ind_tmp))
                data = np.concatenate((data, tmp[row_ind_tmp].toarray().flatten()))

        return c_tmp, col_ind, row_ind, data, len(data)

    def calc_weight_sparse(self,X, Bsp, lam=0.001, seed=0, nmax=100000, b=100):
        n_leaf, d = X.shape
        random.seed(seed)

        c_all = np.zeros((nmax, 1))
        dz = Bsp.shape[0]

        np.random.seed(seed)
        ind1 = np.random.randint(0, n_leaf, nmax)
        ind2 = np.random.randint(0, n_leaf, nmax)


        # Multi proces
        result = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.calc_weight_in)(X, Bsp, ind1[b * i:(i + 1) * b], ind2[b * i:(i + 1) * b]) for i in
            range(int(nmax / b)))

        n_ele = 0
        for ii in range(int(nmax / b)):
            n_ele += result[ii][4]

        col_ind = np.zeros(n_ele)
        row_ind = np.zeros(n_ele)
        data = np.zeros(n_ele)
        st = 0
        ed = result[0][4]
        for ii in range(int(nmax / b) - 1):
            c_all[ii * b:(ii + 1) * b, 0] = result[ii][0]
            col_ind[st:ed] = result[ii][1] + (ii) * b
            row_ind[st:ed] = result[ii][2]
            data[st:ed] = result[ii][3]
            st += result[ii][4]
            ed += result[ii + 1][4]

        ii = int(nmax / b) - 1
        c_all[ii * b:(ii + 1) * b, 0] = result[ii][0]
        col_ind[st:ed] = result[ii][1] + (ii) * b
        row_ind[st:ed] = result[ii][2]
        data[st:ed] = result[ii][3]

        n_sample = nmax
        c = np.asfortranarray(c_all[:n_sample, 0].reshape((n_sample, 1)), dtype='float32')

        Zsp = sparse.csc_matrix((data, (col_ind, row_ind)), shape=(nmax, dz), dtype='float32')

        # Solving nonnegative Lasso
        param = {'numThreads': -1, 'verbose': True,
                 'lambda1': lam, 'it0': 10, 'max_it': 2000, 'tol': 1e-3, 'intercept': False,
                 'pos': True}

        param['loss'] = 'square'
        param['regul'] = 'l1'

        W0 = np.ones((Zsp.shape[1], c.shape[1]), dtype='float32', order="F")

        (W, optim_info) = spams.fistaFlat(c, Zsp, W0, True, **param)
        
        return W



    def pairwiseTWD(self,a,b):
        # Compute the Tree Wasserstein

        TWD = abs(self.wB.dot(a - b)).sum(0) / self.n_slice

        return TWD

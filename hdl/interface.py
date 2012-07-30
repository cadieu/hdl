"""
Specifies a sklearn-like interface for learning sparsecoding models
"""
import numpy as np
from learners import SGD
from models import SparseSlowModel
from display import display_final

class SparseCoding(object):

    def __init__(self,n_atoms, alpha=.1, max_iter=10000, batchsize=32, whitenpatches=40000):
        self.n_atoms = n_atoms
        self.alpha = alpha
        self.max_iter = max_iter
        self.batchsize = batchsize
        self.whitenpatches = whitenpatches
        self._setup_complete = False
        self._learning_complete = False

    @property
    def components_(self):
        if self._learning_complete:
            return np.dot(self._model.dewhitenmatrix,self._model.A.get_value()).T
        else:
            return

    def setup(self,X):

        self._model=SparseSlowModel(patch_sz=None,D=X.shape[1],N=self.n_atoms,T=self.batchsize,
            sparse_cost='l1',slow_cost=None,lam_sparse=self.alpha,perc_var=99.0)

        self._learner = SGD(model=self._model,datasource='X',eta_target_maxupdate=.001,
            batchsize=self.batchsize,save_every=10000,display_every=1000,input_data=X)

        self._setup_complete = True

    def get_learning_schedule(self):

        train_epochs = 8
        anneal_epochs = 2

        iter_chunk = int(np.ceil(self.max_iter*(1./(train_epochs+anneal_epochs))))
        train_iterations = iter_chunk
        anneal_iterations = iter_chunk
        train_list  = [{'iterations':train_iterations} for epoch in range(train_epochs)]
        anneal_list = [{'iterations':anneal_iterations,'change_target':.5} for epoch in range(anneal_epochs)]
        return train_list + anneal_list

    def fit(self,X,y=None):

        if not self._setup_complete: self.setup(X)

        databatch = self._learner.get_databatch(self.whitenpatches)
        self._learner.model.learn_whitening(databatch)
        self._learner.model.setup()

        sched_list = self.get_learning_schedule()

        for sdict in sched_list:
            if sdict.has_key('change_target'):
                self._learner.change_target(sdict['change_target'])
            if sdict.has_key('batchsize'):
                self._learner.batchsize = sdict['batchsize']
            if sdict.has_key('iterations'):
                self._learner.learn(iterations=sdict['iterations'])
            else:
                self._learner.learn()

        display_final(self._model)

        self._learning_complete = True

        return self

    def transform(self,X,y=None,output_function='infer',chunk_size=1000):

        return self._model(X.T,output_function=output_function,chunk_size=chunk_size).T

"""
Specifies a sklearn-like interface for learning sparsecoding models
"""
import numpy as np
from learners import SGD
from models import SparseSlowModel, ConvSparseSlowModel, ConvWhitenInputModel
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

class ConvWhitening(object):
    def __init__(self, imshp, whitenfiltershp=(7, 7), whiten_perc_var=100.,
                 whitenpatches=1000,**kargs):
        self.imshp = imshp
        assert len(self.imshp) == 4
        self.whitenfiltershp = whitenfiltershp
        self.whiten_perc_var = whiten_perc_var
        self.whitenpatches = whitenpatches
        self._setup_complete = False
        self._learning_complete = False

    @property
    def components_(self):
        if self._learning_complete:
            return self._model.convwhitenfilter.reshape((self._model.nchannels**2,-1)).T
        else:
            return

    def setup(self, X):
        self._model = ConvWhitenInputModel(imshp=self.imshp,
            convwhitenfiltershp=self.whitenfiltershp, perc_var=self.whiten_perc_var)

        self._learner = SGD(model=self._model, datasource='X', save_every=10000, display_every=1000, input_data=X)

        self._setup_complete = True

    def fit(self, X, y=None):
        if not self._setup_complete: self.setup(X)

        databatch = self._learner.get_databatch(self.whitenpatches)
        self._learner.model.learn_whitening(databatch)

        display_final(self._model)

        self._learning_complete = True

        return self

    def transform(self, X, y=None):
        return self._model(X.T).T

class ConvSparseCoding(object):

    def __init__(self,imshp, kshp, alpha=1., whiten=True,whitenfiltershp=(7,7),whiten_perc_var=100.,max_iter=10000, batchsize=4, whitenpatches=1000,
                 start_eta_target_maxupdate=.05,strides=(1,1),
                 sparse_cost='l1',slow_cost=None,**kargs):
        self.imshp = imshp
        if len(self.imshp) == 3:
            self.imshp = (batchsize, self.imshp[0], self.imshp[1], self.imshp[2])
        else:
            self.imshp = (batchsize, self.imshp[1], self.imshp[2], self.imshp[3])
        self.kshp = kshp
        self.whiten = whiten
        self.whitenfiltershp = whitenfiltershp
        self.whiten_perc_var = whiten_perc_var
        self.n_atoms = kshp[0]
        self.alpha = alpha
        self.max_iter = max_iter
        self.batchsize = batchsize
        self.strides = strides
        self.sparse_cost = sparse_cost
        self.slow_cost = slow_cost
        self.whitenpatches = whitenpatches
        self.start_eta_target_maxupdate = start_eta_target_maxupdate
        self._setup_complete = False
        self._learning_complete = False
        self.stop_center_basis_functions_epoch = kargs.get('stop_center_basis_functions_epoch',1)

    @property
    def components_(self):
        if self._learning_complete:
            return self._model.A.get_value().T
        else:
            return

    def setup(self,X):

        self._model=ConvSparseSlowModel(imshp=self.imshp,kshp=self.kshp,T=self.batchsize,stride=self.strides,
            sparse_cost=self.sparse_cost,slow_cost=self.slow_cost,lam_sparse=self.alpha,whiten=self.whiten,
            convwhitenfiltershp=self.whitenfiltershp,perc_var=self.whiten_perc_var)

        self._learner = SGD(model=self._model,datasource='X',eta_target_maxupdate=self.start_eta_target_maxupdate,
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

        for sind, sdict in enumerate(sched_list):
            if sdict.has_key('change_target'):
                self._learner.change_target(sdict['change_target'])
            if sdict.has_key('batchsize'):
                self._learner.batchsize = sdict['batchsize']
            if sind == self.stop_center_basis_functions_epoch:
                self._model.center_basis_functions = False
            if sdict.has_key('iterations'):
                self._learner.learn(iterations=sdict['iterations'])
            else:
                self._learner.learn()

        display_final(self._model)

        self._learning_complete = True

        return self

    def transform(self,X,y=None,output_function='infer',chunk_size=1000):

        return self._model(X.T,output_function=output_function,chunk_size=chunk_size).T

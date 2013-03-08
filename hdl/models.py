import os
from copy import copy
import numpy as np
from fista import Fista
import theano

from config import state_dir, verbose, tstring, verbose_timing
from time import time

from scipy import ndimage
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

# Models
class BaseModel(object):

    _params = ['patch_sz','D','tstring','model_name']

    def __init__(self,**kargs):
        self.patch_sz = kargs.get('patch_sz',4)
        self.tstring = kargs.get('tstring', tstring()) # used for saving
        self.model_name = kargs.get('model_name','BaseModel')
        if self.patch_sz:
            self.D = kargs.get('D',self.patch_sz*self.patch_sz)
        else:
            self.D = kargs.get('D',None)

    def __repr__(self):
        s = '='*30 + '\n'
        s += 'Model type: %s'%type(self) + '\n'
        for param in self._params:
            s += '-'*10  + '\n'
            s += '-- ' + param + '\n'
            p = getattr(self,param)
            if hasattr(p,'get_value'):
                p = p.get_value()
            s += str(p) + '\n'
        s += '-'*10 + '\n'

        return s

    def save(self,fname=None,save_name=None,ext='.model',save_txt=False,extra_info=None):
        """save model to disk
        """
        import cPickle
        if save_name is None: save_name = self.model_name
        if fname is None:
            savepath = os.path.join(state_dir,save_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)
            fname = os.path.join(savepath, 'model'+ext)
        else:
            savepath = os.path.split(fname)[0]

        sdict = {}
        for sname in self._params:
            save_prop = getattr(self,sname)
            if hasattr(save_prop,'get_value'): save_prop = save_prop.get_value() # in case we have CudaNdarrayType -> NP
            sdict[sname] = save_prop
        fh = open(fname,'wb')
        cPickle.dump(sdict,fh)
        fh.close()
        if verbose: print "saving model to", fname

        if save_txt:
            repr_string = self.__repr__()
            if not extra_info is None:
                repr_string += '\n'
                if isinstance(extra_info,str):
                    repr_string += extra_info
                elif isinstance(extra_info,list):
                    for extra_item in extra_info:
                        repr_string += str(extra_item)

            model_details_fname = os.path.join(savepath,'model_details.txt')
            with open(model_details_fname,'w') as fh:
                fh.write(repr_string)

        return fname

    def load(self,fname,replace_name=True,reset_theano=True):
        """
        load model from disk

        this load method should be the default load method for the Model class
        (once all models are updated to use the new config objects).
        """
        import cPickle

        noreplace_list = ['model_name','tstring']

        with open(fname,'rb') as fh:
            ldict = cPickle.load(fh)
            for sname in self._params:
                if ldict.has_key(sname):
                    if replace_name:
                        setattr(self,sname,ldict[sname])
                    else:
                        if sname not in noreplace_list:
                            setattr(self,sname,ldict[sname])
                else:
                    print 'WARNING: key %s missing in file %s'%(sname,fname)

        if reset_theano: self._reset_on_load()

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=100,zerophasewhiten=True):
        print 'display not implemented'

    def _reset_on_load(self):
        pass

class WhitenInputModel(BaseModel):
    _params = copy(BaseModel._params)
    _params.extend(['M','whiten','inputmean','whitenmatrix', 'dewhitenmatrix','zerophasewhitenmatrix','num_eigs','perc_var'])

    def __init__(self, **kargs):
        super(WhitenInputModel, self).__init__(**kargs)

        self.whiten = kargs.get('whiten',True)

        if self.whiten:
            self.inputmean = None
            self.whitenmatrix = None
            self.dewhitenmatrix = None
            self.zerophasewhitenmatrix = None
            self.num_eigs = kargs.get('num_eigs',None)
            self.perc_var = kargs.get('perc_var', None)
            if self.num_eigs is None and self.perc_var is None:
                self.perc_var = 99.
            self.M = None
        else:
            self.M = self.D

    def learn_whitening(self,patches):

        if not self.whiten:
            if self.D is None:
                self.D = patches.shape[0]
            return

        from utils import whiten_var

        wpatches, inputmean, whitenmatrix, dewhitenmatrix, zerophasewhitenmatrix, zerophasedewhitenmatrix = whiten_var(patches,num_eigs=self.num_eigs,perc_var=self.perc_var)

        self.inputmean = inputmean
        self.whitenmatrix = whitenmatrix
        self.dewhitenmatrix = dewhitenmatrix
        self.zerophasewhitenmatrix = zerophasewhitenmatrix

        self.M = self.whitenmatrix.shape[0]
        if self.D is None:
            self.D = patches.shape[0]

    def preprocess(self,batch):
        batch -= self.inputmean
        return np.dot(self.whitenmatrix,batch).astype(theano.config.floatX)

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        if not self.whiten:
            return None
        output = {}
        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
        else:
            A = self.whitenmatrix
        if hasattr(A,'get_value'):
            A = A.get_value()

        A = A.T # make the columns the filters

        if self.patch_sz:
            psz = self.patch_sz
        else:
            psz = int(np.ceil(np.sqrt(A.shape[0])))
            if not psz**2 == A.shape[0]:
                A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(A.shape[1],max_factors)
        buf = 1
        sz = int(np.ceil(np.sqrt(NN)))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if Aind >= A.shape[1]: continue
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

class SparseSlowModel(WhitenInputModel):
    """
    Model class for a sparse and slow model
    m = SparseSlowModel(patch_sz, M, N, L, K, rec_cost, sparse_cost, slow_cost)

        patch_sz - image patch side length
        M - size of whitened space
        N - number of subspaces
        T - default temporal length

        rec_cost - reconstruction cost (l2)
        sparse_cost - sparse cost on subspace amplitude (l1)
        slow_cost - cost on time derivative of amplitude (dist)

    """
    _params = copy(WhitenInputModel._params)
    _params.extend(['N','NN','T','A','inference_method','inference_params','lam_sparse','lam_slow','lam_l2','rec_cost','sparse_cost','slow_cost'])

    def __init__(self, **kargs):

        super(SparseSlowModel, self).__init__(**kargs)

        self.N = kargs.get('N',100)
        self.NN = self.N
        self.T = kargs.get('T',128)
        self.rec_cost = kargs.get('rec_cost','l2')
        self.sparse_cost = kargs.get('sparse_cost','l1')
        self.slow_cost = kargs.get('slow_cost','dist')

        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_sparse',.1)))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_slow',.1)))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_l2',.1)))

        self.K = kargs.get('K',8)

        self.inference_method = 'FISTA'
        self.inference_params = {'u_init_method': kargs.get('u_init_method','rand'),
                                 'FISTAargs': {'maxiter': kargs.get('fista_maxiter', 20),
                                               'maxline': kargs.get('fista_maxline', 10),
                                               'errthres': 1e-5,
                                               #'L':1., # default .1
                                               'verbose': False}}

        self.setup_complete = False

        if self.patch_sz:
            self.model_name = kargs.get('model_name','SparseSlowModel_patchsz%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))
        else:
            self.model_name = kargs.get('model_name','SparseSlowModel_N%03d_NN%03d_%s_%s_%s'%(self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))


        if self.slow_cost == 'dist' and self.T < 2:
            raise ValueError, 'self.T must be greater than or equal to 2 with slow_cost = dist'
        #self.setup()

    def _reset_on_load(self):
        if not self.lam_sparse.__class__.__name__ == 'ScalarSharedVariable':
            self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(self.lam_sparse))
        if not self.lam_slow.__class__.__name__ == 'ScalarSharedVariable':
            self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(self.lam_slow))
        if not self.lam_l2.__class__.__name__ == 'ScalarSharedVariable':
            self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(self.lam_l2))
        if not self.A.__class__.__name__ == 'TensorSharedVariable':
            self.A = theano.shared(self.A.astype(theano.config.floatX))

        self.setup(init=False)

    def reset_functions(self):
        self.setup(init=False)

    def setup(self,init=True):
        """Setup model functions"""

        if init:
            A = self.normalize_A(np.random.randn(self.M,self.NN))
            self.A = theano.shared(A.astype(theano.config.floatX))

        # Inference
        if self.rec_cost == 'l2' and self.sparse_cost == 'subspacel1' and self.slow_cost == 'dist':
            print 'Use l2subspacel1slow problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_slow=self.lam_slow,x=self.x,problem_type='l2subspacel1slow')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'subspacel1':
            print 'Use l2subspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,problem_type='l2subspacel1')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'Ksubspacel1':
            print 'Use l2Ksubspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,K=self.K,problem_type='l2Ksubspacel1')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'elastic':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_l2=self.lam_l2,x=self.x,problem_type='l2elastic')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'l1':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x)

        else:
            raise NotImplementedError, 'Inference for rec_cost %s, sparse_cost %s, slow_cost %s'%(self.rec_cost, self.sparse_cost, self.slow_cost)

        # Update gradients:
        if self.rec_cost == 'l2':
            from theano_methods import T_l2_cost, T_l2_cost_norm
            x = theano.tensor.matrix('x')
            u = theano.tensor.matrix('u')
            grad_A = theano.tensor.grad(T_l2_cost_norm(x,u,self.A),self.A)
            self._df_dA = theano.function([x,u],grad_A)
            #grad_A = grad_A.flatten()
            #grad2, _ = theano.scan(fn=lambda i, A, grad_A: theano.tensor.grad(grad_A[i], A).flatten()[i],
            #    sequences=theano.tensor.arange(grad_A.shape[0]), non_sequences=[self.A, grad_A])
            #self._df_dA2 = theano.function([x,u],grad2)
        else:
            raise NotImplementedError, 'rec_cost unknown %s'%self.rec_cost

        self.setup_complete = True

    def gradient(self,batch):

        if not self.setup_complete: raise AssertionError, 'Please call model.setup() before processing'

        t0 = time()
        batch = self.preprocess(batch)
        if verbose_timing: print 'self.preprocess time %f'%(time() - t0)

        t0 = time()
        u = self.inferlatent(batch)
        if verbose_timing: print 'self.inferlatent time %f'%(time() - t0)

        t0 = time()
        modelgradient = self.calc_modelgradient(batch,u)
        if verbose_timing: print 'self.calc_modelgradient time %f'%(time() - t0)

        return modelgradient

    def calc_modelgradient(self,batch,u):

        dA = self._df_dA(batch,u)
        if 'mean' in self.sparse_cost:
            dA[:,-1] = 0.
        grad_dict = dict(dA=dA)

        return grad_dict

    def update_model(self,update_dict,eta=None):
        """ update model parameters with update_dict['dA']
        returns the maximum update percentage max(dA/A)
        """

        A = self.A.get_value()

        A, update_max = self._update_model(A,update_dict,eta=eta)

        A = self.normalize_A(A)
        self.A.set_value(A)

        return update_max

    def _update_model(self,A,update_dict,eta=None):

        param_max = np.max(np.abs(A), axis=0)

        if eta is None:
            update_max = np.max(np.abs(update_dict['dA']), axis=0)
            A -= update_dict['dA']
        else:
            update_max = eta * np.max(np.abs(update_dict['dA']), axis=0)
            A -= eta * update_dict['dA']

        update_max = np.max(update_max / param_max)

        return A, update_max

    def normalize_A(self,A):

        Anorm = np.sqrt((A**2).sum(axis=0)).reshape(1,self.NN)
        return A/Anorm

    def __call__(self,batch,output_function='infer_abs',chunk_size=None):

        return self.output(self.preprocess(batch),
            output_function=output_function,chunk_size=chunk_size)

    def output(self,batch,output_function='infer_abs',chunk_size=None):

        bsz = batch.shape[1]
        if chunk_size is None:
            chunk_size = self.T

        if bsz > chunk_size:
            u_out = None
            done = False
            t0 = 0
            while not done:
                u_minibatch = self._output_minibatch(batch[:,t0:t0+chunk_size],output_function)
                if u_out is None:
                    u_out = np.zeros((u_minibatch.shape[0],bsz))
                u_out[:,t0:t0+chunk_size] = u_minibatch
                t0 += chunk_size
                if t0 >= bsz: done = True
            return u_out
        else:
            return self._output_minibatch(batch,output_function)

    def _output_minibatch(self,batch,output_function):
        if output_function == 'infer':
            return self.inferlatent(batch)
        elif output_function == 'proj':
            return np.dot(self.A.get_value().T,batch)
        elif output_function == 'infer_abs':
            u = self.inferlatent(batch)
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'proj_abs':
            u = np.dot(self.A.get_value().T,batch)
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'infer_loga':
            u = self.inferlatent(batch)
            amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            return np.log(amp + .01)
        elif output_function == 'proj_loga':
            u = np.dot(self.A.get_value().T,batch)
            amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            return np.log(amp + .01)
        elif output_function == 'proj_rect_sat':
            u = np.dot(self.A.get_value().T, batch)
            split_u = np.zeros((u.shape[0] * 2, u.shape[1]), dtype=u.dtype)
            split_u[::2, :] = np.maximum(u, 0.)
            split_u[1::2, :] = np.maximum(-u, 0.)
            # rect:
            rect_value = 8.0
            print 'rect_value:', rect_value, split_u.max()
            return np.minimum(split_u, rect_value)

        else:
            assert NotImplemented, 'Unknown output_function %s'%output_function

    def __initu(self,batch):

        if self.inference_params['u_init_method'] == 'proj':
            return np.dot(self.A.get_value().T,batch)
        elif self.inference_params['u_init_method'] == 'rand':
            return .1*np.random.randn(self.NN, batch.shape[1]).astype(theano.config.floatX)
        else:
            raise NotImplementedError

    def inferlatent(self,batch):
        """

        """
        if not self.inference_method == 'FISTA':
            raise NotImplementedError

        # set shared variables
        t0 = time()
        u0 = self.__initu(batch)
        if verbose_timing: print 'self.__initu time %f'%(time() - t0)

        # run inference
        uhat, fista_history = self._fista(u0, batch, **self.inference_params['FISTAargs'])

        # return
        return uhat

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if self.whiten:
            A = np.dot(self.dewhitenmatrix,A)
            if zerophasewhiten:
                A = np.dot(self.zerophasewhitenmatrix,A)

        if self.patch_sz:
            psz = self.patch_sz
        else:
            psz = int(np.ceil(np.sqrt(A.shape[0])))
            if not psz**2 == A.shape[0]:
                A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(self.NN,max_factors)
        buf = 1
        sz = int(np.ceil(np.sqrt(NN)))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if Aind >= A.shape[1]: continue
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['A'] = array

        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

class BinocColorModel(SparseSlowModel):

    _params = copy(SparseSlowModel._params)
    _params.extend(['binoc_movie_mse_reject'])

    def __init__(self, **kargs):

        patch_sz = kargs.get('patch_sz',8)
        kargs['D'] = 6*patch_sz*patch_sz

        super(BinocColorModel, self).__init__(**kargs)

        self.binoc_movie_mse_reject = kargs.get('binoc_movie_mse_reject',None)

        self.model_name = kargs.get('model_name','BinocColorModel_patchsz%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))

        #self.setup()

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=64,zerophasewhiten=True):
        if not self.whiten:
            return None

        if self.patch_sz is None:
            raise NotImplemented

        from display import display_color_patches

        output = {}
        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
        else:
            A = self.whitenmatrix
        if hasattr(A,'get_value'):
            A = A.get_value()

        A = A.T # make the columns the filters

        # plot the vectors in A
        NN = min(A.shape[1],2*max_factors)
        patches = np.zeros((self.patch_sz,self.patch_sz,3,2*NN))
        count = 0
        for nn in range(NN):
            patches[...,count  ] = A[...,nn].reshape(6,self.patch_sz,self.patch_sz)[:3,...].T
            patches[...,count+1] = A[...,nn].reshape(6,self.patch_sz,self.patch_sz)[3:,...].T
            count += 2

        patches = patches.reshape((self.patch_sz*self.patch_sz*3,2*NN))
        array = display_color_patches(patches,self.patch_sz,fig_num=1,normalize=normalize_A)

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=64,zerophasewhiten=True):

        if self.patch_sz is None:
            raise NotImplemented

        from display import display_color_patches

        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if self.whiten:
            A = np.dot(self.dewhitenmatrix,A)
            if zerophasewhiten:
                A = np.dot(self.zerophasewhitenmatrix,A)

        # plot the vectors in A
        NN = min(self.NN,2*max_factors)

        patches = np.zeros((self.patch_sz,self.patch_sz,3,2*NN))
        count = 0
        for nn in range(NN):
            if normalize_A:
                Ann = A[...,nn]
                Ann /= np.abs(Ann).max() / 127.5
                Ann += 127.5
            else:
                Ann = A[...,nn]
            patches[...,count  ] = Ann.reshape(6,self.patch_sz,self.patch_sz)[:3,...].T
            patches[...,count+1] = Ann.reshape(6,self.patch_sz,self.patch_sz)[3:,...].T
            count += 2

        patches = patches.reshape((self.patch_sz*self.patch_sz*3,2*NN))

        array = display_color_patches(patches,self.patch_sz,fig_num=1,normalize=False)

        output['A'] = array

        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

class HierarchicalModel(object):

    def __init__(self,model_sequence,layer_params):

        self.model_sequence = model_sequence
        self.patch_sz = self.model_sequence[0].patch_sz

        self.layer_params = layer_params

        self.model_name = self.model_sequence[-1].model_name
        self.tstring = self.model_sequence[-1].tstring

    def __call__(self, batch, output_function=None, chunk_size=None):

        for mind, m in enumerate(self.model_sequence):
            if (mind + 1 == len(self.model_sequence)) and (not output_function is None):
                batch = m.output(m.preprocess(batch),output_function,chunk_size=chunk_size)
            else:
                batch = m.output(m.preprocess(batch),self.layer_params[mind]['output_function'],chunk_size=chunk_size)

        return batch

    def __repr__(self):
        return self.model_sequence[-1].__repr__()


class ConvWhitenInputModel(BaseModel):
    _params = copy(BaseModel._params)
    _params.extend(['imshp','nchannels'])
    _params.extend(['M','whiten','convinputmean','convwhitenfilter','convdewhitenfilter','convwhitenfiltershp','inputmean','whitenmatrix', 'dewhitenmatrix','zerophasewhitenmatrix','zerophasedewhitenmatrix','num_eigs','perc_var'])

    def __init__(self, **kargs):
        super(ConvWhitenInputModel, self).__init__(**kargs)

        self.whiten = kargs.get('whiten',True)

        self.imshp = kargs.get('imshp',(32,1,32,32))
        self.nchannels = self.imshp[1]
        self.patch_sz = self.imshp[2]
        assert self.imshp[2] == self.imshp[3], 'only supports square patches now'

        if self.whiten:

            self.convinputmean = None
            self.convwhitenfilter = None
            self.convdewhitenfilter = None
            self.convwhitenfiltershp = kargs.get('convwhitenfiltershp',(5,5))
            assert self.convwhitenfiltershp[0] <= self.imshp[2]
            assert self.convwhitenfiltershp[1] <= self.imshp[3]
            if len(self.convwhitenfiltershp) == 2:
                self.convwhitenfiltershp = (self.nchannels, self.nchannels, self.convwhitenfiltershp[0], self.convwhitenfiltershp[1])
            assert self.convwhitenfiltershp[2]%2, 'Whitening convolution filter should be odd (to pick center tap)'
            assert self.convwhitenfiltershp[3]%2, 'Whitening convolution filter should be odd (to pick center tap)'

            self.inputmean = None
            self.whitenmatrix = None
            self.dewhitenmatrix = None
            self.zerophasewhitenmatrix = None
            self.zerophasedewhitenmatrix = None
            self.num_eigs = kargs.get('num_eigs',None)
            self.perc_var = kargs.get('perc_var', None)
            if self.num_eigs is None and self.perc_var is None:
                self.perc_var = 99.
            self.M = None
        else:
            self.M = self.D
            self.convinputmean = None
            self.convwhitenfilter = None
            self.convwhitenfiltershp = None
            self.convdewhitenfilter = None

            self.inputmean = None
            self.whitenmatrix = None
            self.dewhitenmatrix = None
            self.zerophasewhitenmatrix = None
            self.zerophasedewhitenmatrix = None
            self.num_eigs = None
            self.perc_var = None
            self.M = None

        self.D = np.prod(self.imshp[1:])

    def _reshape_input(self,patches):

        imshp = (patches.shape[1],self.imshp[1],self.imshp[2],self.imshp[3])
        return np.reshape(np.transpose(patches),imshp)

    def _reshape_output(self,output):

        shp = output.shape
        return np.transpose(np.reshape(output,(shp[0],np.prod(shp[1:]))))

    def _convolve4d_view(self,image,kernel,mode='valid',stride=(1,1)):

        from skimage.util.shape import view_as_windows

        imshp = image.shape
        kshp = kernel.shape

        offset = None
        if mode=='valid':
            featshp = (imshp[0],kshp[0],(imshp[2] - kshp[2])/stride[0] + 1,(imshp[3] - kshp[3])/stride[1] + 1) # num images, features, szy, szx
        elif mode == 'same':
            assert stride == (1,1)
            featshp = (imshp[0],kshp[0],imshp[2],imshp[3]) # num images, features, szy, szx
            offset = (kshp[2]/2, kshp[3]/2)
        #elif mode == 'full':
        #    featshp = (imshp[0],kshp[0],imshp[2] + kshp[2] - 1,imshp[3] + kshp[3] - 1) # num images, features, szy, szx
        else:
            raise NotImplemented, 'Unkonwn mode %s'%mode

        kernel_flipped = kernel[:,:,::-1,::-1]

        output = np.zeros(featshp,dtype=image.dtype)
        this_image = None
        for im_i in range(imshp[0]):

            if mode == 'valid':
                this_image = image[im_i,...]
            elif mode == 'same':
                if this_image is None:
                    this_image_shp = (imshp[1], imshp[2] + kshp[2] - 1, imshp[3] + kshp[3] - 1)
                    this_image = np.zeros(this_image_shp,dtype=image.dtype)
                this_image[:,offset[0]:(offset[0]+imshp[2]),offset[1]:(offset[1]+imshp[3])] = image[im_i,...]
            else:
                raise NotImplemented
            imager = view_as_windows(this_image,(kshp[1],kshp[2],kshp[3]))[0,::stride[0],::stride[1],...]
            # imager.shape = (featszr, featszc, channels, ksz[2], ksz[3])
            feat = np.tensordot(kernel_flipped,imager,axes=((1,2,3),(2,3,4)))

            output[im_i,...] = feat

        return output

    def _convolve4d_scipy(self,image,kernel,mode='valid',stride=(1,1),boundary='symm'):

        imshp = image.shape
        kshp = kernel.shape
        if mode=='valid':
            featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx
        elif mode == 'same':
            featshp = (imshp[0],kshp[0],imshp[2],imshp[3]) # num images, features, szy, szx
        elif mode == 'full':
            featshp = (imshp[0],kshp[0],imshp[2] + kshp[2] - 1,imshp[3] + kshp[3] - 1) # num images, features, szy, szx
        else:
            raise NotImplemented, 'Unkonwn mode %s'%mode

        scipy_output = np.zeros(featshp,dtype=image.dtype)
        for im_i in range(imshp[0]):
            for k_i in range(kshp[0]):
                for im_j in range(imshp[1]):
                    scipy_output[im_i,k_i,:,:] += convolve2d(image[im_i,im_j,:,:],kernel[k_i,im_j,:,:],mode=mode,boundary=boundary)

        if not stride==(1,1):
            scipy_output = scipy_output[:,:,::stride[0],::stride[1]]

        return scipy_output

    def learn_whitening(self,patches):

        if not self.whiten:
            if self.D is None:
                self.D = patches.shape[0]
            return

        from utils import whiten_var

        imshp = (patches.shape[1],self.imshp[1],self.imshp[2],self.imshp[3])
        image = self._reshape_input(patches)

        szx,szy = self.convwhitenfiltershp[2:]
        list_patches = []
        for imind in range(imshp[0]):
            for yoffset in range(imshp[2] - szy + 1):
                for xoffset in range(imshp[3] - szx + 1):
                    list_patches.append(image[imind,:,yoffset:yoffset+szy,xoffset:xoffset+szx].ravel())

        array_patches = np.vstack(list_patches).T
        wpatches, inputmean, whitenmatrix, dewhitenmatrix, zerophasewhitenmatrix, zerophasedewhitenmatrix = whiten_var(array_patches,num_eigs=self.num_eigs,perc_var=self.perc_var)

        self.inputmean = inputmean
        self.whitenmatrix = whitenmatrix
        self.dewhitenmatrix = dewhitenmatrix
        self.zerophasewhitenmatrix = zerophasewhitenmatrix
        self.zerophasedewhitenmatrix = zerophasedewhitenmatrix

        # crop out the central tap
        ycnt = int(np.floor(szy/2))
        xcnt = int(np.floor(szx/2))

        # select the inputmean for the central tap for each channel:
        self.convinputmean = np.reshape(inputmean,(self.nchannels,szy,szx))[:,ycnt,xcnt].reshape(1,self.nchannels,1,1)

        # let's do this one step at a time: first reshape
        zerofilters = zerophasewhitenmatrix.reshape(zerophasewhitenmatrix.shape[0],self.nchannels,szy,szx)[:,:,ycnt,xcnt]
        zerodefilters = zerophasedewhitenmatrix.reshape(zerophasedewhitenmatrix.shape[0],self.nchannels,szy,szx)[:,:,ycnt,xcnt]
        # create target array:
        self.convwhitenfilter = np.zeros(self.convwhitenfiltershp)
        self.convdewhitenfilter = np.zeros(self.convwhitenfiltershp)
        # crop out each filter one at a time, and reshape it:
        for nfilt in range(self.convwhitenfiltershp[0]):
            self.convwhitenfilter[nfilt,...] = zerofilters[:,nfilt].reshape(self.nchannels,szy,szx)
            self.convdewhitenfilter[nfilt,...] = zerodefilters[:,nfilt].reshape(self.nchannels,szy,szx)

        if self.D is None:
            self.D = patches.shape[0]

    def __call__(self, batch):
        return self.preprocess(batch)

    def preprocess(self,batch):

        if self.whiten:
            batch = self._reshape_input(batch)

            batch -= self.convinputmean
            output = self._convolve4d_view(batch,self.convwhitenfilter,mode='same')
            output = self._reshape_output(output)
            return output.astype(theano.config.floatX)
        else:
            return batch.astype(theano.config.floatX)

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        if not self.whiten:
            return None

        output = {}

        nfilt, nchannels, szy, szx = self.convwhitenfiltershp

        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
            if hasattr(A,'get_value'):
                A = A.get_value()
            A = A.reshape((A.shape[0]*nchannels,szy*szx))
            Afilter = self.convwhitenfilter.reshape((nfilt*nchannels,szy*szx))
            Adefilter = self.convdewhitenfilter.reshape((nfilt*nchannels,szy*szx))
            A = np.vstack((Afilter,Adefilter,A))
        else:
            A = self.whitenmatrix
            if hasattr(A,'get_value'):
                A = A.get_value()

        A = A.T # make the columns the filters

        psz = int(np.ceil(np.sqrt(A.shape[0])))
        if not psz**2 == A.shape[0]:
            A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(A.shape[1],max_factors)
        buf = 1
        sz = int(np.ceil(np.sqrt(NN)))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if Aind >= A.shape[1]: continue
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

class ConvSparseSlowModel(ConvWhitenInputModel):

    _params = copy(ConvWhitenInputModel._params)
    _params.extend(['N','NN','A','T','K',
                    'kshp','featshp','stride',
                    'center_basis_functions','force_subspace_orthogonal','mask',
                    'inference_method','inference_params',
                    'lam_sparse','lam_slow','lam_l2','rec_cost','sparse_cost','slow_cost'])

    def __init__(self, **kargs):

        super(ConvSparseSlowModel, self).__init__(**kargs)

        self.N = kargs.get('N',8)
        self.NN = self.N
        self.rec_cost = kargs.get('rec_cost','convl2')
        self.sparse_cost = kargs.get('sparse_cost','l1')
        self.slow_cost = kargs.get('slow_cost',None)

        self.T = self.imshp[0]

        self.stride = kargs.get('stride',(1,1))
        if isinstance(self.stride,(int,float)):
            self.stride = (self.stride,self.stride)
        assert isinstance(self.stride,(list,tuple))

        self.mask = kargs.get('mask',True)

        self.kshp = kargs.get('kshp', (self.NN,self.nchannels,8,8))
        if len(self.kshp) == 2:
            self.kshp = (self.NN, self.nchannels, self.kshp[0], self.kshp[1])
        self.featshp = (self.imshp[0],self.kshp[0],
                        (self.imshp[2] - self.kshp[2])/self.stride[0] + 1,
                        (self.imshp[3] - self.kshp[3])/self.stride[1] + 1) # num images, features, szy, szx
        if not self.kshp[0] == self.N or not self.kshp[0] == self.NN:
            self.N = self.kshp[0]
            self.NN = self.N
        self.center_basis_functions = kargs.get('center_basis_functions',True)

        if self.sparse_cost == 'subspacel1':
            assert self.N%2 == 0

        if self.sparse_cost == 'subspacel1mean':
            assert self.N%2 == 1

        self.M = int(np.prod(self.kshp[1:]))

        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_sparse',.1)))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_slow',.1)))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_l2',.1)))

        self.force_subspace_orthogonal = kargs.get('force_subspace_orthogonal',True)
        self.K = kargs.get('K',8)

        self.inference_method = 'FISTA'
        self.inference_params = {'u_init_method': kargs.get('u_init_method','rand'),
                                 'FISTAargs': {'maxiter': kargs.get('fista_maxiter', 20),
                                               'maxline': kargs.get('fista_maxline', 10),
                                               'errthres': 1e-5,
                                               #'L':1., # default .1
                                               'verbose': False}}

        self.setup_complete = False

        if self.patch_sz:
            self.model_name = kargs.get('model_name','ConvSparseSlowModel_patchsz%03d_ksz%03d_nchannels%03d_stride%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.kshp[2], self.nchannels, self.stride[0], self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))
        else:
            self.model_name = kargs.get('model_name','ConvSparseSlowModel_ksz%03d_nchannels%03d_N%03d_NN%03d_%s_%s_%s'%(self.kshp[2], self.nchannels, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))

        if self.slow_cost == 'dist' and self.T < 2:
            raise ValueError, 'self.T must be greater than or equal to 2 with slow_cost = dist'
        #self.setup()

    def _reset_on_load(self):
        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(self.lam_sparse))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(self.lam_slow))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(self.lam_l2))
        self.A = theano.shared(self.A.astype(theano.config.floatX))

        self.setup(init=False)

    def reset_functions(self):
        self.setup(init=False)

    def setup(self,init=True):
        """Setup model functions"""

        if init:
            A = self.normalize_A(np.random.randn(self.M,self.NN))
            self.A = theano.shared(A.astype(theano.config.floatX))

        featdim = int(np.prod(self.featshp[1:]))
        imdim = int(np.prod(self.imshp[1:]))
        # Inference
        if self.rec_cost == 'convl2' and self.sparse_cost == 'subspacel1' and self.slow_cost == 'dist':
            print 'Use l2subspacel1slow problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_slow=self.lam_slow,imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask,x=self.x,problem_type='convl2subspacel1slow')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'subspacel1':
            print 'Use l2subspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask,x=self.x,problem_type='convl2subspacel1')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'subspacel1mean':
            print 'Use l2subspacel1mean problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim, self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u, A=self.A, lam_sparse=self.lam_sparse, imshp=self.imshp, kshp=self.kshp,
                featshp=self.featshp, stride=self.stride, mask=self.mask, x=self.x, problem_type='convl2subspacel1mean')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'Ksubspacel1':
            print 'Use l2Ksubspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask,x=self.x,K=self.K,problem_type='convl2Ksubspacel1')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'elastic':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_l2=self.lam_l2,imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask,x=self.x,problem_type='convl2elastic')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'l1':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(imdim, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(featdim,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask,x=self.x,problem_type='convl2l1')

        else:
            raise NotImplementedError, 'Inference for rec_cost %s, sparse_cost %s, slow_cost %s'%(self.rec_cost, self.sparse_cost, self.slow_cost)

        # Update gradients:
        if self.rec_cost == 'convl2':
            from theano_methods import T_l2_cost_conv
            x = theano.tensor.matrix('x')
            u = theano.tensor.matrix('u')
            if self.stride == (1,1):
                grad_A = theano.tensor.grad(T_l2_cost_conv(x,u,self.A,self.imshp,self.kshp,featshp=self.featshp,stride=self.stride,mask=self.mask),self.A)
                if theano.config.device == 'cpu':
                    self._df_dA = theano.function([x,u],grad_A)
                else:
                    from theano.sandbox.cuda import host_from_gpu
                    self._df_dA = theano.function([x,u],host_from_gpu(grad_A))
            else:
                from theano_methods import T_l2_cost_conv_dA
                grad_A = T_l2_cost_conv_dA(x,u,self.A,
                                           imshp=self.imshp,kshp=self.kshp,featshp=self.featshp,
                                           stride=self.stride,mask=self.mask)
                if theano.config.device == 'cpu':
                    self._df_dA = theano.function(inputs=[x,u],outputs=grad_A)
                else:
                    from theano.sandbox.cuda import host_from_gpu

                    #profmode = theano.ProfileMode(optimizer='fast_run',linker=theano.gof.OpWiseCLinker())
                    #self._df_dA = theano.function(inputs=[x,u],outputs=grad_A,mode=profmode)

                    self._df_dA = theano.function(inputs=[x,u],outputs=grad_A)

                    #self._df_dA = theano.function(inputs=[x,u],outputs=host_from_gpu(grad_A),mode=profmode)
        else:
            raise NotImplementedError, 'rec_cost unknown %s'%self.rec_cost

        self.setup_complete = True

    def gradient(self,batch):

        if not self.setup_complete: raise AssertionError, 'Please call model.setup() before processing'

        t0 = time()
        batch = self.preprocess(batch)
        if verbose_timing: print 'self.preprocess time %f'%(time() - t0)

        t0 = time()
        u = self.inferlatent(batch)
        if verbose_timing: print 'self.inferlatent time %f'%(time() - t0)

        t0 = time()
        modelgradient = self.calc_modelgradient(batch,u)
        if verbose_timing: print 'self.calc_modelgradient time %f'%(time() - t0)

        return modelgradient

    def calc_modelgradient(self,batch,u):

        dA = self._df_dA(batch, u)
        if 'mean' in self.sparse_cost:
            dA[:, -1] = 0.
        grad_dict = dict(dA=dA)

        return grad_dict

    def update_model(self,update_dict,eta=None):
        """ update model parameters with update_dict['dA']
        returns the maximum update percentage max(dA/A)
        """

        A = self.A.get_value()

        A, update_max = self._update_model(A,update_dict,eta=eta)

        A = self.normalize_A(A)
        self.A.set_value(A)

        return update_max

    def _update_model(self,A,update_dict,eta=None):
        param_max = np.max(np.abs(A), axis=0)

        if eta is None:
            update_max = np.max(np.abs(update_dict['dA']), axis=0)
            A -= update_dict['dA']
        else:
            update_max = eta * np.max(np.abs(update_dict['dA']), axis=0)
            A -= eta * update_dict['dA']

        update_max = np.max(update_max / param_max)

        if (self.sparse_cost == 'subspacel1' or self.sparse_cost == 'subspacel1mean')\
        and self.force_subspace_orthogonal:
            if not hasattr(self, '_force_subspace_orthogonal_toggle'):
                self._force_subspace_orthogonal_toggle = True
            if not hasattr(self, '_force_subspace_orthogonal_counter'):
                self._force_subspace_orthogonal_counter = 0
                self._force_subspace_orthogonal_every = 1

            if not self._force_subspace_orthogonal_counter % self._force_subspace_orthogonal_every:
                if self._force_subspace_orthogonal_toggle:
                    for n in range(self.N / 2):
                        factor = np.dot(A[:, 2 * n].T, A[:, 2 * n + 1]) / np.dot(A[:, 2 * n].T, A[:, 2 * n])
                        A[:, 2 * n] = A[:, 2 * n] - factor * A[:, 2 * n + 1]
                else:
                    for n in range(self.N / 2):
                        factor = np.dot(A[:, 2 * n].T, A[:, 2 * n + 1]) / np.dot(A[:, 2 * n + 1].T, A[:, 2 * n + 1])
                        A[:, 2 * n + 1] = A[:, 2 * n + 1] - factor * A[:, 2 * n]
                self._force_subspace_orthogonal_toggle = not self._force_subspace_orthogonal_toggle

            self._force_subspace_orthogonal_counter += 1
            self._force_subspace_orthogonal_counter %= self._force_subspace_orthogonal_every

        return A, update_max

    def normalize_A(self,A):

        if self.center_basis_functions:
            A = np.reshape(A.T,self.kshp)
            Atemp = np.zeros_like(A)
            cm_y0 = self.kshp[2]/2
            cm_x0 = self.kshp[3]/2
            for n in range(A.shape[0]):
                im = np.sum(np.abs(A[n,...]),axis=0)
                cm_y, cm_x = np.round(ndimage.measurements.center_of_mass(im))
                cm_y = int(cm_y)
                cm_x = int(cm_x)
                if cm_y > cm_y0:
                    y_delta = cm_y - cm_y0
                    y0slice = slice(y_delta,self.kshp[2])
                    yslice = slice(0,self.kshp[2] - y_delta)
                else:
                    y_delta = cm_y0 - cm_y
                    y0slice = slice(0,self.kshp[2] - y_delta)
                    yslice = slice(y_delta,self.kshp[2])
                if cm_x > cm_x0:
                    x_delta = cm_x - cm_x0
                    x0slice = slice(x_delta,self.kshp[2])
                    xslice = slice(0,self.kshp[2] - x_delta)
                else:
                    x_delta = cm_x0 - cm_x
                    x0slice = slice(0,self.kshp[2] - x_delta)
                    xslice = slice(x_delta,self.kshp[2])

                Atemp[n,:,yslice,xslice] = A[n,:,y0slice,x0slice]
            A = np.reshape(Atemp.T,(np.prod(self.kshp[1:]),self.kshp[0]))

        if 'mean' in self.sparse_cost:
            A[:,-1] = 1.

        Anorm = np.sqrt((A**2).sum(axis=0)).reshape(1,self.NN)
        return A/Anorm

    def __call__(self,batch,output_function='infer_abs',chunk_size=None):

        return self.output(self.preprocess(batch),
            output_function=output_function,chunk_size=chunk_size)

    def output(self,batch,output_function='infer_abs',chunk_size=None):

        if 'reweight' in output_function: self._calc_reweight_features(dtype=batch.dtype)

        bsz = batch.shape[1]
        if chunk_size is None:
            chunk_size = self.T

        if bsz > chunk_size:
            u_out = None
            done = False
            t0 = 0
            while not done:
                u_minibatch = self._output_minibatch(batch[:,t0:t0+chunk_size],output_function)
                if u_out is None:
                    u_out = np.zeros((u_minibatch.shape[0],bsz))
                u_out[:,t0:t0+chunk_size] = u_minibatch
                t0 += chunk_size
                if t0 >= bsz: done = True
            return u_out
        else:
            return self._output_minibatch(batch,output_function)

    def _output_minibatch(self,batch,output_function):
        if output_function == 'infer':
            return self.inferlatent(batch)
        elif output_function == 'proj':
            kernel = np.reshape(self.A.get_value().T,self.kshp)
            image = np.reshape(batch.T,self.imshp)
            features = self._convolve4d_view(image,kernel,stride=self.stride)
            features = features[:,:,:self.featshp[2],:self.featshp[3]]
            reshaped_features = np.transpose(np.reshape(features,(self.featshp[0],np.prod(self.featshp[1:]))))
            return reshaped_features
        elif output_function == 'infer_abs':
            u = self.inferlatent(batch)
            return self._abs(u)
        elif output_function == 'proj_abs':
            kernel = np.reshape(self.A.get_value().T,self.kshp)
            image = np.reshape(batch.T,self.imshp)
            features = self._convolve4d_view(image,kernel,stride=self.stride)
            features = features[:,:,:self.featshp[2],:self.featshp[3]]
            u = np.transpose(np.reshape(features,(self.featshp[0],np.prod(self.featshp[1:]))))
            return self._abs(u)
        elif output_function == 'infer_loga':
            u = self.inferlatent(batch)
            amp = self._abs(u)
            return np.log(amp + .01)
        elif output_function == 'proj_loga':
            kernel = np.reshape(self.A.get_value().T,self.kshp)
            image = np.reshape(batch.T,self.imshp)
            features = self._convolve4d_view(image,kernel,stride=self.stride)
            features = features[:,:,:self.featshp[2],:self.featshp[3]]
            u = np.transpose(np.reshape(features,(self.featshp[0],np.prod(self.featshp[1:]))))
            amp = self._abs(u)
            return np.log(amp + .01)
        elif output_function == 'infer_abs_reweight':
            u = self.inferlatent(batch)
            features = np.reshape(u.T,self.featshp)
            features = self._reweight_features(features)
            u = np.transpose(np.reshape(features,(self.featshp[0],np.prod(self.featshp[1:]))))
            return self._abs(u)
        elif output_function == 'proj_abs_reweight':
            kernel = np.reshape(self.A.get_value().T,self.kshp)
            image = np.reshape(batch.T,self.imshp)
            features = self._convolve4d_view(image,kernel,stride=self.stride)
            features = features[:,:,:self.featshp[2],:self.featshp[3]]
            features = self._reweight_features(features)
            u = np.transpose(np.reshape(features,(self.featshp[0],np.prod(self.featshp[1:]))))
            return self._abs(u)
        else:
            assert NotImplemented, 'Unknown output_function %s'%output_function

    def _abs(self,u):
        if 'mean' in self.sparse_cost:
            u = np.reshape(np.transpose(u),self.featshp)
            a = np.sqrt(u[:,:-1:2,:,:]**2 + u[:,1:-1:2,:,:]**2)
            m = u[:,-1,:,:][:,None,:,:]
            new_feat = np.concatenate((a,m),axis=1)
            return np.transpose(np.reshape(new_feat,(self.featshp[0],-1)))
        else:
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)

    def _calc_reweight_features(self,dtype):
        kernel = np.reshape(self.A.get_value().T,self.kshp)
        feature_weights = np.zeros((1, self.featshp[1], 1, 1), dtype=dtype)
        if self.convdewhitenfilter is None:
            feature_weights[:] = 1.
        else:
            deweighted_kernel = self._convolve4d_scipy(kernel, self.convdewhitenfilter, mode='same', boundary='fill')
            #new_kernel = np.reshape(Aoutput, (self.kshp[0], np.prod(self.kshp[1:]))).T
            for n in range(self.featshp[1]):
                feature_weights[0, n, 0, 0] = np.sqrt(np.sum(deweighted_kernel[n, :, :, :] ** 2))
            feature_weights /= np.max(feature_weights)
            #print 'feature_weights.min()', feature_weights.min()
        self._feature_weights = feature_weights

    def _reweight_features(self,features):

        output_features = features*self._feature_weights
        return output_features

    def __initu(self,batch):

        if self.inference_params['u_init_method'] == 'proj':
            raise NotImplemented
            #return np.dot(self.A.get_value().T,batch)
        elif self.inference_params['u_init_method'] == 'rand':
            return .001*np.random.randn(np.prod(self.featshp[1:]), batch.shape[1]).astype(theano.config.floatX)
        else:
            raise NotImplementedError

    def inferlatent(self,batch):
        """

        """
        if not self.inference_method == 'FISTA':
            raise NotImplementedError

        # set shared variables
        t0 = time()
        u0 = self.__initu(batch)
        if verbose_timing: print 'self.__initu time %f'%(time() - t0)

        # run inference
        uhat, fista_history = self._fista(u0, batch, **self.inference_params['FISTAargs'])

        # return
        return uhat

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if not zerophasewhiten and self.whiten:
            Areshaped = np.reshape(A.T,self.kshp)
            Aoutput = self._convolve4d_scipy(Areshaped,self.convdewhitenfilter,mode='same',boundary='fill')
            A = np.reshape(Aoutput,(self.kshp[0],np.prod(self.kshp[1:]))).T

        psz = int(np.ceil(np.sqrt(A.shape[0])))
        if not psz**2 == A.shape[0]:
            A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(self.NN,max_factors)
        buf = 1
        sz = int(np.ceil(np.sqrt(NN)))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if Aind >= A.shape[1]: continue
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['A'] = array

        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

